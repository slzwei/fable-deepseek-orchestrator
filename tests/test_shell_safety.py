"""No unsafe shell construction anywhere in the shipped source."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
MODULES = sorted(SRC.rglob("*.py"))
assert MODULES, "no source modules found"


def parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_eval_exec_or_system(path):
    banned = {"eval", "exec", "compile", "execfile"}
    banned_attrs = {"system", "popen", "spawnl", "spawnv", "spawnve"}
    for node in ast.walk(parsed(path)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in banned:
                raise AssertionError(f"{path.name}:{node.lineno} calls {func.id}()")
            if isinstance(func, ast.Attribute) and func.attr in banned_attrs:
                owner = getattr(func.value, "id", "")
                if owner in ("os", "subprocess", "commands"):
                    raise AssertionError(
                        f"{path.name}:{node.lineno} calls {owner}.{func.attr}()")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_subprocess_is_never_given_a_shell(path):
    for node in ast.walk(parsed(path)):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "shell":
                value = keyword.value
                is_false = isinstance(value, ast.Constant) and value.value is False
                assert is_false, f"{path.name}:{node.lineno} passes shell={ast.dump(value)}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_subprocess_arguments_are_lists_not_strings(path):
    """A string first argument to Popen/run is a command line, which we never build."""
    for node in ast.walk(parsed(path)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in ("run", "Popen", "check_output", "call", "check_call"):
            continue
        owner = getattr(getattr(func, "value", None), "id", "")
        if owner not in ("subprocess", ""):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            raise AssertionError(
                f"{path.name}:{node.lineno} passes a command string to {name}()")
        if isinstance(first, ast.JoinedStr):
            raise AssertionError(
                f"{path.name}:{node.lineno} passes an f-string command to {name}()")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_shell_string_helpers_are_used_to_build_commands(path):
    """shlex.split on model text would reintroduce command construction."""
    source = path.read_text(encoding="utf-8")
    assert "shlex.split" not in source, f"{path.name} uses shlex.split"
    assert "os.system" not in source, f"{path.name} references os.system"


def test_the_only_process_launcher_is_the_hardened_runner():
    """Direct subprocess use is confined to a few audited, argv-only places."""
    allowed = {"runner.py", "workspace.py", "context.py", "orchestrator.py"}
    offenders = []
    for path in MODULES:
        source = path.read_text(encoding="utf-8")
        if "subprocess." in source and path.name not in allowed:
            offenders.append(path.name)
    assert not offenders, f"unaudited subprocess use in: {offenders}"


def test_the_runner_always_disables_the_shell():
    source = (SRC / "fabds" / "runner.py").read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "start_new_session=True" in source, "timeouts must kill the whole group"
    assert "timeout" in source


def test_command_arguments_are_never_interpolated_from_model_text():
    """The command action carries an id; nothing builds argv from model strings."""
    source = (SRC / "fabds" / "workers.py").read_text(encoding="utf-8")
    assert 'payload["command_id"]' in source
    assert "assert_command" in source
    for forbidden in ('payload["command"]', 'payload["argv"]', 'payload["cmd"]'):
        assert forbidden not in source, f"workers.py reads {forbidden} from model output"


def test_model_supplied_paths_always_pass_through_containment():
    source = (SRC / "fabds" / "workers.py").read_text(encoding="utf-8")
    assert source.count("assert_read") >= 3
    assert source.count("assert_write") >= 2
