"""DeepSeek off-peak scheduling.

Peak is 01:00-04:00 and 06:00-10:00 UTC on working weekdays; everything else is
off-peak at half rate. The gate must never let a prompt out at peak rates when
it is armed, must always terminate, and must be overridable.

Every test drives an injected clock, so none of them waits on a real one.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from fabds.errors import PeakHoursBlocked
from fabds.pricing import (
    describe,
    is_peak,
    next_offpeak_start,
    parse_holidays,
    seconds_until_offpeak,
    status,
)
from fabds.providers.base import CompletionRequest
from fabds.providers.deepseek_http import DeepSeekHttpProvider


def utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


# -- window classification --------------------------------------------------

@pytest.mark.parametrize("moment,peak", [
    ("2026-09-22T00:59", False),  # Tue, just before window 1
    ("2026-09-22T01:00", True),   # window 1 opens
    ("2026-09-22T03:59", True),   # still window 1
    ("2026-09-22T04:00", False),  # window 1 closes (end exclusive)
    ("2026-09-22T05:30", False),  # gap between windows
    ("2026-09-22T06:00", True),   # window 2 opens
    ("2026-09-22T09:59", True),   # still window 2
    ("2026-09-22T10:00", False),  # window 2 closes
    ("2026-09-22T18:00", False),  # evening
    ("2026-09-22T23:59", False),  # night
])
def test_peak_windows(moment, peak):
    assert is_peak(utc(moment)) is peak


@pytest.mark.parametrize("moment", [
    "2026-09-26T02:00",  # Saturday
    "2026-09-27T07:00",  # Sunday
])
def test_weekends_are_entirely_off_peak(moment):
    assert is_peak(utc(moment)) is False
    assert seconds_until_offpeak(utc(moment)) == 0


def test_listed_holidays_are_off_peak():
    """Chinese public holidays are off-peak in full, but only if declared."""
    moment = utc("2026-10-01T02:00")  # a weekday inside window 1
    assert is_peak(moment) is True, "undeclared: assumed a working day"
    holidays = parse_holidays(["2026-10-01"])
    assert is_peak(moment, holidays=holidays) is False


def test_holiday_assumption_errs_toward_waiting_not_overspending():
    """Guessing wrong must cost time, never money.

    An undeclared holiday is treated as a working day, so fabds may wait when it
    did not need to. The opposite error - treating a working day as a holiday -
    would send traffic at peak rates believing it was cheap.
    """
    moment = utc("2026-10-01T02:00")
    assert is_peak(moment) is True


def test_naive_datetimes_are_rejected():
    with pytest.raises(ValueError, match="naive"):
        is_peak(datetime(2026, 9, 22, 2, 0))


# -- next off-peak instant --------------------------------------------------

def test_next_offpeak_clears_the_window_with_a_margin():
    """Exiting exactly at 04:00:00 races the boundary; the margin avoids that."""
    target = next_offpeak_start(utc("2026-09-22T02:00"), safety_margin_s=60)
    assert target == utc("2026-09-22T04:01")


def test_next_offpeak_is_a_noop_when_already_off_peak():
    moment = utc("2026-09-22T23:00")
    assert next_offpeak_start(moment) == moment


def test_worst_case_wait_is_hours_not_overnight():
    """Peak is the narrow window, so waiting is bounded and cheap."""
    worst = max(seconds_until_offpeak(utc(f"2026-09-22T{h:02d}:{m:02d}"))
                for h in range(24) for m in (0, 30))
    assert worst <= 4.1 * 3600, f"worst-case wait was {worst / 3600:.1f}h"


def test_describe_is_readable():
    assert "PEAK now" in describe(utc("2026-09-22T02:00"))
    assert "off-peak now" in describe(utc("2026-09-22T23:00"))


# -- the gate ---------------------------------------------------------------

PEAK = datetime(2026, 9, 22, 2, 0, tzinfo=timezone.utc)
OFFPEAK = datetime(2026, 9, 22, 23, 0, tzinfo=timezone.utc)


def provider(config, *, clock, sleep, **overrides):
    return DeepSeekHttpProvider(replace(config, **overrides), sleep=sleep, clock=clock)


def test_gate_is_off_by_default(config):
    """Default behaviour is unchanged: no waiting, no surprise."""
    slept = []
    provider(config, clock=lambda: PEAK, sleep=slept.append)._await_offpeak()
    assert slept == []


def test_gate_waits_during_peak_and_resumes_past_the_margin(config):
    clock = {"t": PEAK}
    instance = provider(
        config, clock=lambda: clock["t"],
        sleep=lambda s: clock.__setitem__("t", clock["t"] + timedelta(seconds=s)),
        deepseek_offpeak_only=True, deepseek_offpeak_max_wait_s=5 * 3600,
    )
    instance._await_offpeak()
    assert clock["t"] >= utc("2026-09-22T04:01"), "must clear the window plus margin"
    assert not is_peak(clock["t"])


def test_gate_does_not_wait_when_already_off_peak(config):
    slept = []
    provider(config, clock=lambda: OFFPEAK, sleep=slept.append,
             deepseek_offpeak_only=True)._await_offpeak()
    assert slept == []


def test_override_proceeds_immediately_during_peak(config):
    """--peak-ok is the user's explicit decision to pay full rate."""
    slept = []
    provider(config, clock=lambda: PEAK, sleep=slept.append,
             deepseek_offpeak_only=True, deepseek_allow_peak=True)._await_offpeak()
    assert slept == []


def test_override_beats_the_config_file(config):
    """A config that arms the gate must still yield to an explicit --peak-ok."""
    armed = replace(config, deepseek_offpeak_only=True)
    slept = []
    provider(armed, clock=lambda: PEAK, sleep=slept.append,
             deepseek_allow_peak=True)._await_offpeak()
    assert slept == []


def test_a_predicted_wait_over_the_limit_is_refused_not_slept(config):
    with pytest.raises(PeakHoursBlocked, match="exceeds"):
        provider(config, clock=lambda: PEAK, sleep=lambda _s: None,
                 deepseek_offpeak_only=True,
                 deepseek_offpeak_max_wait_s=300)._await_offpeak()


def test_a_clock_that_never_advances_terminates(config):
    """Spinning forever is a worse failure than paying peak rates: it is silent.

    The limit here clears the pre-check, so this exercises the loop's own
    bounds - real elapsed time and an iteration cap - not the pre-check.
    """
    slept = []
    with pytest.raises(PeakHoursBlocked, match="gave up waiting"):
        provider(config, clock=lambda: PEAK, sleep=slept.append,
                 deepseek_offpeak_only=True,
                 deepseek_offpeak_max_wait_s=5 * 3600)._await_offpeak()
    assert slept, "it should have tried to wait before giving up"


def test_the_gate_sits_at_the_transport_boundary(config, monkeypatch):
    """A library caller that skips the orchestrator must still be gated."""
    instance = provider(config, clock=lambda: PEAK, sleep=lambda _s: None,
                        deepseek_offpeak_only=True,
                        deepseek_offpeak_max_wait_s=60)
    sent = []
    monkeypatch.setattr(instance._opener, "open",
                        lambda *a, **k: sent.append(1))
    with pytest.raises(PeakHoursBlocked):
        instance.complete(CompletionRequest("s", "u", "deepseek-flash"))
    assert sent == [], "no request may reach the wire during peak while armed"


def test_wait_is_announced_rather_than_silent(config):
    seen = []
    instance = provider(config, clock=lambda: PEAK, sleep=lambda _s: None,
                        deepseek_offpeak_only=True,
                        deepseek_offpeak_max_wait_s=5 * 3600)
    instance.on_wait = seen.append
    with pytest.raises(PeakHoursBlocked):
        instance._await_offpeak()
    assert seen and seen[0].peak, "a multi-hour hold must be reported, not silent"


def test_provider_status_reports_the_window(config):
    armed = replace(config, deepseek_offpeak_only=True)
    armed.deepseek_api_key_file.write_text("x" * 20, encoding="utf-8")
    detail = DeepSeekHttpProvider(armed).status().detail
    assert "off-peak gate on" in detail

    overridden = replace(armed, deepseek_allow_peak=True)
    assert "OVERRIDDEN" in DeepSeekHttpProvider(overridden).status().detail


def test_cli_flags_map_to_config():
    from fabds.cli import build_parser

    parsed = build_parser().parse_args(["run", "task", "--offpeak"])
    assert parsed.offpeak is True and parsed.peak_ok is False
    parsed = build_parser().parse_args(["run", "task", "--offpeak", "--peak-ok"])
    assert parsed.offpeak is True and parsed.peak_ok is True


def test_bad_holiday_dates_are_a_config_error(config):
    from fabds.errors import ConfigError

    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        replace(config, offpeak_extra_dates=("not-a-date",)).validate()
