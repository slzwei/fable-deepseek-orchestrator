"""A fake MCP server used to *prove* isolation rather than assert it.

The probe speaks just enough MCP to be accepted by a client, and appends a line
to a marker file the moment it starts. Both ``fabds doctor`` and the regression
suite use it the same way:

1. register the probe in a throwaway ``HOME`` and launch the CLI normally
   -> the marker must appear, otherwise the probe itself is broken and the
   test proves nothing;
2. launch the CLI the way fabds launches it -> the marker must NOT appear;
3. launch with the probe granted explicitly -> the marker must appear again,
   showing that isolation is a policy we apply, not a capability we lack.

Step 1 is what makes the result meaningful. Without it, "no marker" could
simply mean the probe never worked.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["write_probe", "probe_config", "IsolationProbeResult", "SERVER_SOURCE"]

SERVER_SOURCE = '''\
#!/usr/bin/env python3
"""Fake MCP server. Records that it started, then answers the handshake."""
import json, os, pathlib, sys

marker = pathlib.Path(os.environ.get("FABDS_PROBE_MARKER", "/tmp/fabds-probe.marker"))
try:
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("a", encoding="utf-8") as fh:
        fh.write("STARTED\\n")
except OSError:
    pass

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        message = json.loads(line)
    except ValueError:
        continue
    message_id = message.get("id")
    if message_id is None:
        continue
    method = message.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "fabds-probe", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [{
            "name": "fabds_probe_tool",
            "description": "probe",
            "inputSchema": {"type": "object"},
        }]}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result}) + "\\n")
    sys.stdout.flush()
'''


@dataclass
class IsolationProbeResult:
    control_started: bool
    isolated_started: bool
    granted_started: bool | None = None
    detail: str = ""

    @property
    def meaningful(self) -> bool:
        """The test only proves something if the probe demonstrably works."""
        return self.control_started

    @property
    def passed(self) -> bool:
        return self.meaningful and not self.isolated_started

    def describe(self) -> str:
        if not self.meaningful:
            return "inconclusive: the probe never started even without isolation"
        if self.isolated_started:
            return "FAILED: an unrelated MCP server initialised inside an isolated call"
        if self.granted_started is False:
            return "passed (note: an explicitly granted server also failed to start)"
        return "passed: unrelated MCP servers do not initialise"


def write_probe(directory: Path, marker: Path) -> Path:
    """Write the probe script and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fabds_mcp_probe.py"
    script.write_text(SERVER_SOURCE, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def probe_config(script: Path, marker: Path, *, name: str = "fabds_probe") -> dict:
    """An MCP server entry that runs the probe with a given marker path."""
    return {
        "mcpServers": {
            name: {
                "command": sys.executable,
                "args": [str(script)],
                "env": {"FABDS_PROBE_MARKER": str(marker)},
            }
        }
    }


def write_fake_home(home: Path, script: Path, marker: Path) -> Path:
    """A throwaway HOME whose ``.claude.json`` registers the probe."""
    home = Path(home)
    (home / ".claude").mkdir(parents=True, exist_ok=True)
    config = {"hasCompletedOnboarding": True, **probe_config(script, marker)}
    (home / ".claude.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Copy credentials in when they exist so the control run behaves normally.
    # Absence is fine: MCP servers start during session init, before auth matters.
    source = Path(os.path.expanduser("~")) / ".claude" / "oauth_tokens.json"
    if source.is_file():
        try:
            (home / ".claude" / "oauth_tokens.json").write_bytes(source.read_bytes())
        except OSError:
            pass
    return home / ".claude.json"
