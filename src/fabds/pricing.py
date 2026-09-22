"""DeepSeek off-peak scheduling.

DeepSeek bills peak and off-peak differently, and off-peak is half price:

    Peak      01:00-04:00 and 06:00-10:00 UTC, Monday to Friday,
              excluding Chinese public holidays
    Off-peak  everything else, including weekends and Chinese public
              holidays in full

Source: https://api-docs.deepseek.com/quick_start/pricing/

So peak is the *narrow* window. Waiting for off-peak costs at most about four
hours, never overnight, and halves the bill for every worker token.

Two design decisions worth stating plainly:

**Conservative about holidays.** Chinese public holidays are off-peak in full,
but this module has no reliable holiday calendar and will not invent one. It
therefore assumes a weekday is a working day unless the operator lists the date
in ``offpeak_extra_dates``. The error is always in the safe direction: we may
wait when we did not have to, which costs time, never money. Guessing the other
way would send traffic at peak rates believing it was cheap.

**Boundaries are approached with a margin.** A request fired at 03:59:59 is
billed however the server classifies it on arrival. The gate therefore waits
until the peak window has ended *plus* a small margin rather than racing it.

Everything here is a pure function of an aware ``datetime``, so the behaviour is
deterministic and fully testable without waiting for a clock.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

__all__ = [
    "PEAK_WINDOWS_UTC", "PeakStatus", "is_peak", "next_offpeak_start",
    "seconds_until_offpeak", "describe",
]

#: (start, end) pairs in UTC. End is exclusive.
PEAK_WINDOWS_UTC: tuple[tuple[time, time], ...] = (
    (time(1, 0), time(4, 0)),
    (time(6, 0), time(10, 0)),
)

DEFAULT_SAFETY_MARGIN_S = 60


class PeakStatus:
    """The result of classifying one instant."""

    def __init__(self, peak: bool, window: "tuple[time, time] | None",
                 next_offpeak: datetime, now: datetime) -> None:
        self.peak = peak
        self.window = window
        self.next_offpeak = next_offpeak
        self.now = now

    @property
    def wait_seconds(self) -> float:
        return max(0.0, (self.next_offpeak - self.now).total_seconds())

    def as_dict(self) -> dict:
        return {
            "peak": self.peak,
            "now_utc": self.now.isoformat(),
            "window_utc": (f"{self.window[0]:%H:%M}-{self.window[1]:%H:%M}"
                           if self.window else None),
            "next_offpeak_utc": self.next_offpeak.isoformat(),
            "wait_seconds": round(self.wait_seconds, 1),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PeakStatus(peak={self.peak}, wait={self.wait_seconds:.0f}s)"


def _as_utc(moment: datetime | None) -> datetime:
    if moment is None:
        return datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError("a naive datetime is ambiguous; pass an aware one")
    return moment.astimezone(timezone.utc)


def _is_working_day(day: date, holidays: "frozenset[date] | None") -> bool:
    if day.weekday() >= 5:              # Saturday, Sunday
        return False
    if holidays and day in holidays:    # operator-supplied Chinese holidays
        return False
    return True


def is_peak(moment: datetime | None = None, *,
            holidays: "frozenset[date] | None" = None) -> bool:
    """True when ``moment`` falls inside a DeepSeek peak window."""
    moment = _as_utc(moment)
    if not _is_working_day(moment.date(), holidays):
        return False
    current = moment.timetz().replace(tzinfo=None)
    return any(start <= current < end for start, end in PEAK_WINDOWS_UTC)


def _current_window(moment: datetime) -> "tuple[time, time] | None":
    current = moment.timetz().replace(tzinfo=None)
    for start, end in PEAK_WINDOWS_UTC:
        if start <= current < end:
            return (start, end)
    return None


def next_offpeak_start(moment: datetime | None = None, *,
                       holidays: "frozenset[date] | None" = None,
                       safety_margin_s: int = DEFAULT_SAFETY_MARGIN_S) -> datetime:
    """The next instant that is safely off-peak.

    Returns ``moment`` itself when it is already off-peak, so callers can treat
    this as "wait until", with zero wait being the common case.
    """
    moment = _as_utc(moment)
    if not is_peak(moment, holidays=holidays):
        return moment

    window = _current_window(moment)
    assert window is not None  # is_peak() was true, so we are inside one
    end = datetime.combine(moment.date(), window[1], tzinfo=timezone.utc)
    candidate = end + timedelta(seconds=safety_margin_s)

    # The margin can land inside the *next* peak window only if two windows are
    # adjacent, which they are not today. Re-check anyway so a future change to
    # PEAK_WINDOWS_UTC cannot silently produce a peak-time wake-up.
    if is_peak(candidate, holidays=holidays):
        return next_offpeak_start(candidate, holidays=holidays,
                                  safety_margin_s=safety_margin_s)
    return candidate


def status(moment: datetime | None = None, *,
           holidays: "frozenset[date] | None" = None,
           safety_margin_s: int = DEFAULT_SAFETY_MARGIN_S) -> PeakStatus:
    moment = _as_utc(moment)
    return PeakStatus(
        peak=is_peak(moment, holidays=holidays),
        window=_current_window(moment) if is_peak(moment, holidays=holidays) else None,
        next_offpeak=next_offpeak_start(moment, holidays=holidays,
                                        safety_margin_s=safety_margin_s),
        now=moment,
    )


def seconds_until_offpeak(moment: datetime | None = None, *,
                          holidays: "frozenset[date] | None" = None,
                          safety_margin_s: int = DEFAULT_SAFETY_MARGIN_S) -> float:
    return status(moment, holidays=holidays, safety_margin_s=safety_margin_s).wait_seconds


def describe(moment: datetime | None = None, *,
             holidays: "frozenset[date] | None" = None) -> str:
    """A one-line human summary, for logs and ``doctor``."""
    state = status(moment, holidays=holidays)
    if not state.peak:
        return (f"off-peak now ({state.now:%Y-%m-%d %H:%M}Z); "
                "DeepSeek is billing at half the peak rate")
    wait = state.wait_seconds
    return (f"PEAK now ({state.now:%Y-%m-%d %H:%M}Z, window "
            f"{state.window[0]:%H:%M}-{state.window[1]:%H:%M} UTC); "
            f"off-peak resumes {state.next_offpeak:%H:%M}Z "
            f"in {wait / 3600:.1f}h")


def parse_holidays(values) -> "frozenset[date]":
    """Parse ``YYYY-MM-DD`` strings from configuration."""
    parsed = set()
    for value in values or ():
        parsed.add(date.fromisoformat(str(value)))
    return frozenset(parsed)
