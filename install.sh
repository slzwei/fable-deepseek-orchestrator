#!/usr/bin/env bash
#
# Install the fable-deepseek-orchestrator skill.
#
# What this does: copies a fixed list of files from this checkout into
# ~/.codex/skills/fable-deepseek-orchestrator/ and writes two small launchers.
#
# What this deliberately does not do: download anything, install a package,
# require sudo, register a LaunchAgent or cron job, start a daemon, or set up
# any kind of auto-update. Read it before you run it; it is meant to be short
# enough to read.
#
# Usage:
#   ./install.sh                 install to ~/.codex/skills
#   ./install.sh --prefix DIR    install somewhere else
#   ./install.sh --dry-run       print what would happen, change nothing
#   ./install.sh --uninstall     remove a previous installation
#   ./install.sh --check         verify an installation is complete

set -euo pipefail

SKILL_NAME="fable-deepseek-orchestrator"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${FABDS_PREFIX:-$HOME/.codex/skills}"
MODE="install"

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)    PREFIX="$2"; shift 2 ;;
    --prefix=*)  PREFIX="${1#*=}"; shift ;;
    --dry-run)   MODE="dry-run"; shift ;;
    --uninstall) MODE="uninstall"; shift ;;
    --check)     MODE="check"; shift ;;
    -h|--help)   sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

TARGET="$PREFIX/$SKILL_NAME"

# The exact set of files this installs. Nothing is globbed from outside it.
PAYLOAD=(
  "skill/SKILL.md:SKILL.md"
  "skill/agents/openai.yaml:agents/openai.yaml"
  "skill/references/task-contract.md:references/task-contract.md"
  "skill/references/escalation-policy.md:references/escalation-policy.md"
  "skill/references/security-boundaries.md:references/security-boundaries.md"
  "README.md:README.md"
  "SECURITY.md:SECURITY.md"
  "VERSION:VERSION"
)

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

if [ "$(id -u)" = "0" ]; then
  die "refusing to run as root; this installs into your own home directory only"
fi

PYTHON="${FABDS_PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found on PATH"
"$PYTHON" - <<'PY' || die "Python 3.10 or newer is required"
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY

case "$MODE" in
  uninstall)
    if [ -d "$TARGET" ]; then
      rm -rf "$TARGET"
      say "removed $TARGET"
    else
      say "nothing to remove at $TARGET"
    fi
    say "note: caches under ~/.cache/fabds and per-repository .fabds/ directories are left alone."
    say "      remove them yourself with: rm -rf ~/.cache/fabds"
    exit 0
    ;;
  check)
    [ -d "$TARGET" ] || die "not installed at $TARGET"
    missing=0
    for entry in "${PAYLOAD[@]}"; do
      dest="${entry#*:}"
      [ -f "$TARGET/$dest" ] || { say "missing: $dest"; missing=1; }
    done
    [ -x "$TARGET/scripts/orchestrate" ] || { say "missing: scripts/orchestrate"; missing=1; }
    [ -x "$TARGET/scripts/doctor" ] || { say "missing: scripts/doctor"; missing=1; }
    [ -f "$TARGET/orchestrator/fabds/cli.py" ] || { say "missing: orchestrator/fabds/cli.py"; missing=1; }
    [ "$missing" = "0" ] || exit 1
    say "installation at $TARGET is complete ($(cat "$TARGET/VERSION"))"
    exit 0
    ;;
esac

say "fable-deepseek-orchestrator $(cat "$SOURCE_DIR/VERSION")"
say "  source: $SOURCE_DIR"
say "  target: $TARGET"
say ""

if [ "$MODE" = "dry-run" ]; then
  say "would create $TARGET and copy:"
  for entry in "${PAYLOAD[@]}"; do say "  ${entry%%:*}  ->  ${entry#*:}"; done
  say "  src/fabds/**.py  ->  orchestrator/fabds/"
  say "would write launchers: scripts/orchestrate, scripts/doctor"
  say "nothing was changed."
  exit 0
fi

for entry in "${PAYLOAD[@]}"; do
  src="${entry%%:*}"
  [ -f "$SOURCE_DIR/$src" ] || die "missing source file: $src (is this a complete checkout?)"
done

mkdir -p "$TARGET/scripts" "$TARGET/agents" "$TARGET/references" "$TARGET/orchestrator"

for entry in "${PAYLOAD[@]}"; do
  src="$SOURCE_DIR/${entry%%:*}"
  dest="$TARGET/${entry#*:}"
  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
  say "  installed ${entry#*:}"
done

# The Python package. Copied wholesale, but only .py files, and only from src/.
rm -rf "$TARGET/orchestrator/fabds"
mkdir -p "$TARGET/orchestrator/fabds/providers"
for file in "$SOURCE_DIR"/src/fabds/*.py; do
  cp "$file" "$TARGET/orchestrator/fabds/"
done
for file in "$SOURCE_DIR"/src/fabds/providers/*.py; do
  cp "$file" "$TARGET/orchestrator/fabds/providers/"
done
cp "$SOURCE_DIR/VERSION" "$TARGET/orchestrator/VERSION"
say "  installed orchestrator/fabds ($(ls "$TARGET/orchestrator/fabds"/*.py | wc -l | tr -d ' ') modules)"

cat > "$TARGET/scripts/orchestrate" <<'LAUNCHER'
#!/usr/bin/env bash
# Thin launcher. All logic lives in the Python package; this only sets the path.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$HERE/orchestrator${PYTHONPATH:+:$PYTHONPATH}"
exec "${FABDS_PYTHON:-python3}" -m fabds "$@"
LAUNCHER

cat > "$TARGET/scripts/doctor" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$HERE/orchestrator${PYTHONPATH:+:$PYTHONPATH}"
exec "${FABDS_PYTHON:-python3}" -m fabds doctor "$@"
LAUNCHER

chmod 755 "$TARGET/scripts/orchestrate" "$TARGET/scripts/doctor"
say "  installed scripts/orchestrate, scripts/doctor"

say ""
say "installed. next:"
say "  $TARGET/scripts/doctor"
say "  $TARGET/scripts/orchestrate run 'your task' --dry-run"
say ""
say "no background service, no scheduled job and no auto-update were installed."
say "to remove: $SOURCE_DIR/install.sh --uninstall"
