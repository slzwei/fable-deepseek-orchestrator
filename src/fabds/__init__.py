"""fabds - Fable/DeepSeek orchestration controller.

Role split enforced by this package:

* **Controller** (Codex/Astra, the caller) keeps final authority. This package
  never merges work into a user branch on its own and never grants a model a
  capability the controller did not write into a work packet.
* **Planner** (Fable 5.1) is consulted for architecture and critique only. It
  runs with no tools, no MCP servers and no settings inheritance.
* **Workers** (DeepSeek V4.1 Flash) execute bounded packets through a typed
  action protocol. They never receive a shell.

Everything here is Python standard library only.
"""

from pathlib import Path

__all__ = ["__version__", "VERSION_FILE"]

def _version_candidates() -> list[Path]:
    """VERSION sits beside the checkout in development and beside the package
    once installed, so both layouts are checked."""
    here = Path(__file__).resolve()
    return [
        here.parents[2] / "VERSION",   # <repo>/VERSION with src/fabds/
        here.parents[1] / "VERSION",   # <skill>/orchestrator/VERSION
        here.parent / "VERSION",
    ]


VERSION_FILE = _version_candidates()[0]


def _read_version() -> str:
    for candidate in _version_candidates():
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text
    return "0.0.0+unknown"


__version__ = _read_version()
