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

VERSION_FILE = Path(__file__).resolve().parents[2] / "VERSION"


def _read_version() -> str:
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0+unknown"


__version__ = _read_version()
