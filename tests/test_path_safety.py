"""Path containment and the worker permission boundary."""

from __future__ import annotations

import pathlib

import pytest

from fabds.errors import PathSafetyError, PermissionDeniedError
from fabds.pathsafety import assert_relative_path, is_within, resolve_within
from fabds.permissions import (
    CommandSpec,
    WorkerPermissions,
    assert_not_globally_forbidden,
    check_command_spec,
)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    (root / "src" / "parser").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "parser" / "core.py").write_text("x = 1\n", encoding="utf-8")
    (root / "tests" / "test_core.py").write_text("pass\n", encoding="utf-8")
    (root / "unrelated.py").write_text("y = 2\n", encoding="utf-8")
    return root


TRAVERSAL = [
    "../escape.txt", "../../etc/passwd", "a/../../b", "./../x",
    "/etc/passwd", "/tmp/anything", "~/.ssh/id_rsa", "~root/.bashrc",
    "src/../../outside.py",
]


@pytest.mark.parametrize("candidate", TRAVERSAL)
def test_traversal_and_absolute_paths_are_rejected(workspace, candidate):
    assert is_within(workspace, candidate) is False
    with pytest.raises(PathSafetyError):
        resolve_within(workspace, candidate)


def test_dot_dot_slash_filter_bypass_is_not_a_traversal_here(workspace):
    """``....//....//x`` defeats filters that strip "../" by substring.

    fabds does not filter by substring: it resolves the path for real. Under
    POSIX ``....`` is an ordinary directory name, so this payload correctly
    stays *inside* the workspace rather than escaping it. Asserting that keeps
    a future refactor from reintroducing string-stripping logic.
    """
    resolved = resolve_within(workspace, "....//....//etc/passwd")
    assert str(resolved).startswith(str(workspace.resolve()))
    assert resolved != pathlib.Path("/etc/passwd")


def test_ordinary_relative_paths_are_accepted(workspace):
    assert resolve_within(workspace, "src/parser/core.py").is_file()
    assert resolve_within(workspace, "new/nested/file.txt").parent.name == "nested"


def test_nul_byte_is_rejected(workspace):
    with pytest.raises(PathSafetyError, match="NUL"):
        resolve_within(workspace, "src/core\x00.py")


def test_symlink_escape_is_rejected(workspace, tmp_path):
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("classified", encoding="utf-8")
    (workspace / "link.txt").symlink_to(secret)
    with pytest.raises(PathSafetyError, match="escapes the workspace"):
        resolve_within(workspace, "link.txt")


def test_symlinked_directory_escape_is_rejected(workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("classified", encoding="utf-8")
    (workspace / "shortcut").symlink_to(outside)
    with pytest.raises(PathSafetyError, match="escapes the workspace"):
        resolve_within(workspace, "shortcut/secret.txt")


def test_internal_symlinks_still_work(workspace):
    (workspace / "alias.py").symlink_to(workspace / "src" / "parser" / "core.py")
    assert resolve_within(workspace, "alias.py").name == "core.py"


def test_writes_refuse_to_follow_a_final_symlink(workspace, tmp_path):
    (workspace / "inner.txt").write_text("a", encoding="utf-8")
    (workspace / "alias.txt").symlink_to(workspace / "inner.txt")
    resolve_within(workspace, "alias.txt")  # reading through it is fine
    with pytest.raises(PathSafetyError, match="write through a symlink"):
        resolve_within(workspace, "alias.txt", follow_final_symlink=False)


def test_symlinked_workspace_root_is_supported(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "file.txt").write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(real)
    assert resolve_within(link, "file.txt").is_file()


@pytest.mark.parametrize("raw", ["", "   ", "/abs", "~/home", "a/../b", "\x00"])
def test_shape_validation_rejects_bad_input(raw):
    with pytest.raises(PathSafetyError):
        assert_relative_path(raw)


# -- ownership --------------------------------------------------------------

def test_worker_cannot_write_outside_ownership(workspace):
    permissions = WorkerPermissions(workspace, owned=("src/parser/**",), readonly=("tests/**",))
    permissions.assert_write("src/parser/core.py")
    for denied in ("unrelated.py", "tests/test_core.py", "src/other.py"):
        with pytest.raises(PermissionDeniedError, match="not owned"):
            permissions.assert_write(denied)


def test_read_only_worker_cannot_write_anything(workspace):
    permissions = WorkerPermissions(workspace, readonly=("**",), read_only=True)
    permissions.assert_read("src/parser/core.py")
    with pytest.raises(PermissionDeniedError, match="read-only"):
        permissions.assert_write("src/parser/core.py")


def test_read_only_worker_cannot_be_given_owned_paths(workspace):
    from fabds.errors import ConfigError

    with pytest.raises(ConfigError):
        WorkerPermissions(workspace, owned=("src/**",), read_only=True)


def test_reads_outside_the_grant_are_denied(workspace):
    permissions = WorkerPermissions(workspace, owned=("src/parser/**",), readonly=("tests/**",))
    permissions.assert_read("tests/test_core.py")
    with pytest.raises(PermissionDeniedError, match="read denied"):
        permissions.assert_read("unrelated.py")


def test_credential_paths_are_denied_even_when_owned(workspace):
    (workspace / "src" / "parser" / ".env").write_text("K=v\n", encoding="utf-8")
    permissions = WorkerPermissions(workspace, owned=("src/**",))
    with pytest.raises(PermissionDeniedError):
        permissions.assert_read("src/parser/.env")
    with pytest.raises(PermissionDeniedError):
        permissions.assert_write("src/parser/.env")


def test_global_forbidden_roots_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".ssh").mkdir()
    with pytest.raises(PermissionDeniedError, match="globally forbidden"):
        assert_not_globally_forbidden(tmp_path / ".ssh")
    with pytest.raises(PermissionDeniedError, match="globally forbidden"):
        assert_not_globally_forbidden(tmp_path / ".ssh" / "id_rsa")
    with pytest.raises(PermissionDeniedError):
        WorkerPermissions(workspace_root=tmp_path / ".claude")
    assert_not_globally_forbidden(tmp_path / "projects" / "app")  # must not raise


# -- commands ---------------------------------------------------------------

FORBIDDEN_COMMANDS = [
    ["sudo", "rm", "-rf", "/"],
    ["ssh", "host", "whoami"],
    ["curl", "https://attacker.invalid"],
    ["bash", "-c", "echo hi"],
    ["sh", "-c", "echo hi"],
    ["launchctl", "load", "x.plist"],
    ["git", "push", "origin", "main"],
    ["git", "remote", "add", "evil", "url"],
    ["npm", "install", "-g", "evil"],
    ["pip", "install", "--global", "evil"],
    ["/usr/bin/sudo", "ls"],
    ["aws", "s3", "sync", ".", "s3://x"],
    ["kubectl", "delete", "pod", "--all"],
]


@pytest.mark.parametrize("argv", FORBIDDEN_COMMANDS)
def test_dangerous_commands_are_refused(argv):
    with pytest.raises((PermissionDeniedError, Exception)):
        check_command_spec(argv)


@pytest.mark.parametrize("argv", [
    ["python3", "-m", "pytest", "-q"],
    ["make", "test"],
    ["git", "status", "--porcelain"],
    ["npm", "test", "--silent"],
])
def test_ordinary_commands_are_allowed(argv):
    check_command_spec(argv)


def test_shell_syntax_in_argv_is_rejected_loudly():
    from fabds.errors import ConfigError

    with pytest.raises(ConfigError, match="shell syntax"):
        check_command_spec(["python3", "-m", "pytest", "|", "tee", "log"])
    with pytest.raises(ConfigError, match="shell syntax"):
        check_command_spec(["echo", "$(whoami)"])


def test_a_worker_selects_a_command_it_cannot_compose(workspace):
    spec = CommandSpec("pytest", ("python3", "-m", "pytest", "-q"), max_extra_paths=1)
    permissions = WorkerPermissions(workspace, owned=("src/**",), commands={"pytest": spec})

    assert permissions.assert_command("pytest") is spec
    with pytest.raises(PermissionDeniedError, match="not in this worker's allowlist"):
        permissions.assert_command("rm_rf_slash")

    # Extra arguments are paths, validated against the workspace, count-capped.
    assert spec.build_argv(["src/parser"], workspace)[-1] == "src/parser"
    with pytest.raises(PathSafetyError):
        spec.build_argv(["../../etc/passwd"], workspace)
    with pytest.raises(PermissionDeniedError, match="at most 1"):
        spec.build_argv(["src", "tests"], workspace)


def test_command_spec_argv_is_frozen_at_construction():
    spec = CommandSpec("t", ("python3", "-m", "pytest"))
    assert isinstance(spec.argv, tuple)
    with pytest.raises(Exception):
        spec.argv = ("rm", "-rf", "/")  # frozen dataclass
