# fable-deepseek-orchestrator

A Codex skill that puts three models in their right places:

- **Fable 5.1** thinks about architecture. It has no tools and cannot touch
  anything. Its output is advice.
- **DeepSeek V4.1 Flash** does the volume work — implementing, testing,
  searching, auditing — in parallel, each worker in its own throwaway git
  worktree, each confined to the paths it was granted.
- **Codex/Astra** stays in charge. It authorises what workers may do, verifies
  results against its own evidence, and decides what gets integrated.

Roughly 3,600 lines of Python, standard library only, no daemon, no
auto-update, and a test suite that proves the security properties rather than
asserting them.

```
you
 └─ controller ──► Fable 5.1            architecture + decomposition (advisory)
        │
        ├─ authorises packets           commands and path grants come from YOU
        │
        ├─ DeepSeek worker 01 ──► worktree A   src/parser/**
        ├─ DeepSeek worker 02 ──► worktree B   tests/parser/**
        └─ DeepSeek worker 03 ──► read-only    design review
                │
        ┌───────┘
        ├─ controller re-runs the validation commands itself
        ├─ Fable critique               only when a trigger fires
        └─ staged patches ──► you integrate the ones you accept
```

## Why this exists

Most "orchestrator" scripts have the same three problems. This one addresses
each directly.

**They cannot tell you which model answered.** A `claude` command exits zero and
the output gets labelled "Fable". Here, model identity comes from provider
transport metadata — the CLI's `modelUsage` block, the DeepSeek response's
`model` field — and is re-checked on every call. Requesting Fable and receiving
Opus is a hard error, not a silent substitution.

This matters more than it sounds. Asked to identify itself, the real
`deepseek-flash` endpoint replies *"I'm Claude 3.5 Sonnet, made by Anthropic."*
Model self-report is worthless as evidence. Only the transport knows.

**They hand models a shell.** Here workers emit typed JSON actions and the
controller executes them. `run_command` carries an **id** from an allowlist the
controller wrote; there is no code path anywhere that turns model text into a
shell string, because there is no shell.

**They leak whatever is lying around.** Planner calls start with an empty MCP
configuration, no settings inheritance, no tools, a scrubbed environment and an
empty working directory. `fabds doctor` proves it by starting a real fake MCP
server and confirming it does not appear.

## Install

No `curl | bash`, no sudo, no daemon, no scheduled job, no auto-update.

```bash
git clone <this repo> && cd fable-ds
./install.sh --dry-run     # read what it will do
./install.sh               # copies files into ~/.codex/skills/
~/.codex/skills/fable-deepseek-orchestrator/scripts/doctor
```

Requirements: Python 3.10+, `git`, the `claude` CLI signed in with Fable access,
and a DeepSeek API key in a file. Nothing else — the runtime imports only the
standard library. `pytest` is needed to run the test suite and nothing else;
`./install.sh --uninstall` removes the installation.

Point it at your DeepSeek key with either:

```bash
export DEEPSEEK_API_KEY_FILE=~/path/to/ds-api-key
# or in ~/.config/fabds/config.toml:
#   deepseek_api_key_file = "~/path/to/ds-api-key"
```

The key is read from the file at call time, registered with the redactor the
moment it is read, and never placed in argv, a log or a cache.

## Use

```bash
SKILL=~/.codex/skills/fable-deepseek-orchestrator

$SKILL/scripts/doctor                              # environment + security checks
$SKILL/scripts/orchestrate resolve                 # which models, with evidence
$SKILL/scripts/orchestrate run "<task>" --dry-run  # call nothing, change nothing
$SKILL/scripts/orchestrate run "Add retry with backoff to the HTTP client" \
    --constraints "no new dependencies" --focus src/http/client.py
$SKILL/scripts/orchestrate integrate w01 w03       # apply what you accepted
```

`resolve` shows the resolution and the evidence behind it:

```
Fable 5.1: claude-fable-5-1 via claude_cli (registry)
    evidence: found in the installed CLI binary at .../claude.exe
DeepSeek V4.1 Flash: deepseek-flash via deepseek_http (api)
    evidence: listed by https://api.deepseek.com/models (owned_by=deepseek)
```

Those ids were *discovered*, not guessed: the Claude ids are read out of the
installed CLI's own embedded model table, the DeepSeek ids come from its
`/models` endpoint. If Fable is not present, the run refuses to start.

### A run, end to end

1. Both models resolve, or the run aborts.
2. A context packet is built: repository tree, git state, the files you
   `--focus`, the command allowlist. Credential paths are excluded by name,
   surviving secrets are redacted, and the whole thing is budgeted — sections
   that do not fit are dropped and listed rather than silently truncated.
3. Fable proposes a decomposition. The proposal is *advice*: the controller
   applies its own policy, attaches the commands, clamps the counts, forces
   read-only kinds to be read-only, and turns any ownership overlap into a
   dependency so conflicting packets serialise instead of racing.
4. Workers run under three brakes at once: a global concurrency cap, a per-kind
   cap, and a per-path lock.
5. The controller re-runs each packet's validation commands itself. A worker's
   `tests_passed: true` is a claim; the recorded exit code is evidence.
6. A second Fable call happens only if a named trigger fires — contradicted
   reports, failing validation, repeated failed fixes, several workers hitting
   permission boundaries. Otherwise the controller continues alone.
7. Results are written to `.fabds/runs/<run_id>/` as packets, envelopes and
   patches. **Nothing is merged.** You run `integrate` on what you accept.

### Configuration

`~/.config/fabds/config.toml`, overridden per repository by `.fabds/config.toml`:

```toml
deepseek_api_key_file = "~/quantlab/ds-api-key"
plan_cache_ttl_s = 21600

[limits]
max_workers = 4
max_implementation_workers = 2
max_research_workers = 3
max_planner_rounds = 2
max_total_tasks = 12
max_retries_per_task = 2
max_worker_turns = 12
worker_timeout_s = 600

[models]
# Off by default. Turning it on is recorded in every run's metadata and marked
# [EXPLICIT FALLBACK] in the output. There is no way to make fabds pick a
# substitute model for you.
allow_model_fallback = false
planner_fallback = []

# Commands offered to workers by id. Models select an id; they never compose an
# argv. Policy is checked at load time: sudo, ssh, curl, git push, global
# installs and similar are refused here, not at call time.
[[commands]]
id = "pytest"
argv = ["python3", "-m", "pytest", "-q"]
description = "run the test suite"
timeout_s = 900
max_extra_paths = 1
```

Unknown keys are an error, not a shrug — a typo in a limit fails loudly instead
of silently keeping the default.

## Security boundaries

| | Controller | Planner (Fable) | Worker (DeepSeek) |
|---|---|---|---|
| Read repository | yes | only the packet it is sent | inside its grant |
| Write files | yes | **no** | `owned_paths`, inside its worktree |
| Run commands | yes | **no** | allowlisted ids only |
| MCP servers | yours | **none** unless granted | none; no MCP surface exists |
| Create workers | yes | no | no |
| Integrate | yes | no | no |

Always refused, whatever a packet says: `sudo`, `ssh`, `curl`, `nc`,
`launchctl`; `git push`/`remote`/`config`; global package installs; `~/.ssh`,
`~/.aws`, `~/.claude`, `~/.codex`, keychains, LaunchAgents, `/etc`; and reading
`.env`, `*.pem`, `id_rsa*`, `.netrc`, `.git-credentials` and friends even when
they sit inside an owned glob.

Full threat model in [SECURITY.md](SECURITY.md).

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m pytest              # fast, no network, no spend
.venv/bin/python -m pytest -m live      # real providers, costs a few cents
```

The suite is written to *demonstrate* rather than assert. The MCP isolation test
starts a real fake MCP server three times — registered normally (it must start),
isolated (it must not), explicitly granted (it must start again). The control
run is what makes the negative meaningful: without it, "no marker" could just
mean a broken probe.

The prompt-injection tests assume the model is fully compromised and check that
it achieves nothing anyway.

Two real defects were found by this suite while it was being written: a
model-supplied regex could backtrack forever inside a single `re.search` call
where no in-process timeout can reach it (scanning now runs in a killable child
process), and `Workspace.setup()` failed to snapshot a baseline, which would
have defeated the "claimed success but changed nothing" check.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Required planner model Fable 5.1 could not be resolved` | The CLI does not expose a Fable id. Check `claude --version` and your plan. fabds will not substitute. |
| `rate limited or out of quota` on the planner | Your Fable allowance is exhausted. Wait for the window to reset, or run `--no-planner` and decompose yourself. |
| `model_attestation_failed` | A different model served the request. Investigate; do not work around it. |
| `no DeepSeek key file configured` | Set `DEEPSEEK_API_KEY_FILE` or `deepseek_api_key_file`. |
| Workers keep hitting `denied_actions` | The ownership map is wrong. Widen `owned_paths` deliberately or re-split the packets. |
| `no_changes` on a completed worker | The worker reported success but changed nothing. Usually the objective was unclear or it spent its turns reading. |
| `MCP isolation WARN inconclusive` | The probe could not start even unisolated, so the result proves nothing. Check the `claude` CLI. |
| Patches will not apply | The repository moved since the run. Re-run, or apply by hand. |

## Limitations

- **Fable quota is a hard dependency.** When it is exhausted the planner fails
  closed. That is the design, but it means a run can be blocked by something
  outside the tool. `--no-planner` still works.
- **Workers are not interactive.** No debugger, no REPL, no long-running server.
  A packet needing those belongs to you.
- **Patch integration is textual.** Two packets editing nearby lines in the same
  file will conflict, and resolving it is your job.
- **Non-git targets get copy workspaces**, which are slower and capped at 2,000
  files.
- **The controller verifies, it does not judge.** It can tell you a test failed;
  deciding whether a diff is *good* is still yours.
- **Prompts leave your machine.** Do not orchestrate over a repository you would
  not send to Anthropic and DeepSeek.
- **Benchmark numbers are indicative.** See `bench/README.md` for what was
  measured and what was replayed.
