"""Installation is simple, auditable, reversible and inert."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALL = REPO / "install.sh"


def run_install(*args, home: Path, check=True):
    env = {
        "HOME": str(home), "PATH": os.environ["PATH"],
        "FABDS_PYTHON": sys.executable,
    }
    return subprocess.run(  # noqa: S603
        ["bash", str(INSTALL), *args],
        capture_output=True, text=True, timeout=180, check=check, env=env, cwd=str(REPO),
    )


@pytest.fixture
def fake_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return home


def test_installs_into_a_fake_home(fake_home):
    result = run_install(home=fake_home)
    target = fake_home / ".codex" / "skills" / "fable-deepseek-orchestrator"
    assert target.is_dir()
    for expected in ("SKILL.md", "README.md", "SECURITY.md", "VERSION",
                     "agents/openai.yaml", "references/task-contract.md",
                     "references/escalation-policy.md",
                     "references/security-boundaries.md",
                     "scripts/orchestrate", "scripts/doctor",
                     "orchestrator/fabds/cli.py", "orchestrator/fabds/providers/base.py"):
        assert (target / expected).exists(), f"{expected} was not installed"
    assert os.access(target / "scripts" / "orchestrate", os.X_OK)
    assert "no background service" in result.stdout


def test_the_installed_copy_actually_runs(fake_home):
    run_install(home=fake_home)
    target = fake_home / ".codex" / "skills" / "fable-deepseek-orchestrator"
    result = subprocess.run(  # noqa: S603
        ["bash", str(target / "scripts" / "orchestrate"), "--version"],
        capture_output=True, text=True, timeout=120,
        env={"HOME": str(fake_home), "PATH": os.environ["PATH"], "FABDS_PYTHON": sys.executable},
    )
    assert result.returncode == 0, result.stderr
    assert (REPO / "VERSION").read_text(encoding="utf-8").strip() in result.stdout


def test_check_mode_verifies_a_complete_installation(fake_home):
    run_install(home=fake_home)
    assert run_install("--check", home=fake_home).returncode == 0
    target = fake_home / ".codex" / "skills" / "fable-deepseek-orchestrator"
    (target / "SKILL.md").unlink()
    assert run_install("--check", home=fake_home, check=False).returncode != 0


def test_dry_run_changes_nothing(fake_home):
    result = run_install("--dry-run", home=fake_home)
    assert "nothing was changed" in result.stdout
    assert not (fake_home / ".codex").exists()


def test_uninstall_removes_everything_it_installed(fake_home):
    run_install(home=fake_home)
    target = fake_home / ".codex" / "skills" / "fable-deepseek-orchestrator"
    assert target.is_dir()
    result = run_install("--uninstall", home=fake_home)
    assert not target.exists()
    assert "left alone" in result.stdout, "it should say what it did not remove"


def test_install_is_idempotent(fake_home):
    run_install(home=fake_home)
    target = fake_home / ".codex" / "skills" / "fable-deepseek-orchestrator"
    (target / "orchestrator" / "fabds" / "stale.py").write_text("# stale\n", encoding="utf-8")
    run_install(home=fake_home)
    assert (target / "SKILL.md").exists()
    assert not (target / "orchestrator" / "fabds" / "stale.py").exists(), (
        "reinstalling must replace the package, not merge into it")


def test_a_custom_prefix_is_honoured(fake_home, tmp_path):
    prefix = tmp_path / "elsewhere"
    run_install("--prefix", str(prefix), home=fake_home)
    assert (prefix / "fable-deepseek-orchestrator" / "SKILL.md").is_file()
    assert not (fake_home / ".codex").exists()


def test_installer_performs_no_network_or_privileged_actions():
    """Read the script and assert the absence of the usual footguns."""
    source = INSTALL.read_text(encoding="utf-8")
    for forbidden in ("curl", "wget", "pip install", "npm install", "brew install",
                      "launchctl", "crontab", "systemctl", "LaunchAgents",
                      "nohup", "&  #", "git clone", "sudo "):
        assert forbidden not in source.replace(
            "require sudo, register a LaunchAgent or cron job", ""
        ).replace("no sudo", ""), f"installer must not contain {forbidden!r}"
    assert "set -euo pipefail" in source


def test_installer_refuses_to_run_as_root():
    source = INSTALL.read_text(encoding="utf-8")
    assert 'if [ "$(id -u)" = "0" ]' in source
    assert "refusing to run as root" in source


def test_no_auto_update_mechanism_exists():
    """Nothing in the shipped tree fetches or updates itself."""
    shipped = list((REPO / "src").rglob("*.py")) + [INSTALL]
    for path in shipped:
        text = path.read_text(encoding="utf-8")
        for forbidden in ("urlretrieve", "git pull", "pip install", "self_update",
                          "auto_update", "check_for_updates"):
            assert forbidden not in text, f"{path.name} contains {forbidden!r}"


def test_only_the_deepseek_endpoint_is_ever_contacted():
    """The one outbound host is the configured DeepSeek base URL."""
    import re

    hosts: set[str] = set()
    for path in (REPO / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # Host patterns inside the redaction regexes are escaped (hooks\.slack\.com),
        # so normalise before extracting or they truncate at the first escape.
        text = text.replace("\\.", ".")
        hosts |= set(re.findall(r"https?://([a-zA-Z0-9._-]+)", text))
    allowed = {
        "api.deepseek.com",        # the only endpoint fabds actually calls
        "hooks.slack.com",         # appears only inside a redaction pattern
        "attacker.invalid",        # appears only in a docstring about proxies
    }
    assert hosts <= allowed, f"unexpected outbound hosts referenced: {hosts - allowed}"


def test_package_has_no_third_party_imports():
    """Runtime is standard library only, so installation cannot pull anything in."""
    import ast

    stdlib = set(sys.stdlib_module_names)
    for path in (REPO / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [] if node.level else [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                assert name in stdlib or name == "fabds", (
                    f"{path.name} imports third-party module {name!r}")


def test_installs_for_claude_code_as_well(fake_home):
    """The same skill works from ~/.claude/skills; the frontmatter format is shared."""
    run_install("--claude", home=fake_home)
    target = fake_home / ".claude" / "skills" / "fable-deepseek-orchestrator"
    assert (target / "SKILL.md").is_file()
    assert os.access(target / "scripts" / "orchestrate", os.X_OK)
    # The examples must point at where it actually landed, not the Codex path.
    skill = (target / "SKILL.md").read_text(encoding="utf-8")
    assert f"SKILL={target}" in skill
    assert ".codex/skills/fable-deepseek-orchestrator\n" not in skill
    assert not (target / "SKILL.md.bak").exists(), "sed backup must be cleaned up"


def test_all_installs_to_both_hosts(fake_home):
    run_install("--all", home=fake_home)
    for host in (".codex", ".claude"):
        target = fake_home / host / "skills" / "fable-deepseek-orchestrator"
        assert (target / "orchestrator" / "fabds" / "cli.py").is_file(), host
        assert f"SKILL={target}" in (target / "SKILL.md").read_text(encoding="utf-8")
