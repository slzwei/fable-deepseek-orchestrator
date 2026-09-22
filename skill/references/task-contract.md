# Work packet and result envelope contract

## Work packet

Everything a worker knows. There is no conversation history, no controller
reasoning, and no repository dump — only this.

| Field | Meaning |
|---|---|
| `task_id` | lowercase `[a-z0-9_-]`, unique in the run; also the worktree and patch name |
| `kind` | `research`, `implement`, `test`, `audit`, `document`. `research` and `audit` are read-only and cannot own paths |
| `objective` | one sentence. If it needs "and", consider two packets |
| `context` | what this worker needs to know that it cannot discover. Not a repository dump |
| `owned_paths` | globs it may write. Empty for read-only kinds |
| `readonly_paths` | globs it may read. Empty means the whole workspace |
| `forbidden_paths` | explicit denials on top of the always-on credential exclusions |
| `acceptance_criteria` | statements that can be *checked*, not judged |
| `commands` | the allowlist, as complete argv arrays the controller wrote |
| `validation_command_ids` | the subset the worker must run before finishing |
| `depends_on` | task ids that must succeed first |
| `max_turns` | hard cap on model turns |

### Ownership rules

- Two packets that are meant to run in parallel must not share an `owned_paths`
  glob. Overlaps are detected and converted into a dependency, which serialises
  them — correct, but it silently costs you the parallelism.
- High-contention files (`package.json`, lockfiles, migrations, central schemas)
  belong to exactly one packet, or to the controller.
- A packet that needs to read a file another packet owns should list it under
  `readonly_paths` **and** declare it in `depends_on`. The dependency is what
  actually matters: a dependent packet's workspace is seeded with its
  dependencies' completed results, so without it the packet inspects a tree
  that does not contain the work yet. Test and audit packets almost always
  want this.

## The action protocol

One JSON object per turn: `{"actions": [...]}`, at most four actions.

| Action | Fields | Authorisation |
|---|---|---|
| `read_file` | `path` | inside the workspace, readable, not a credential path |
| `list_dir` | `path` | as above; credential files are listed but marked excluded |
| `search` | `pattern`, optional `path` | runs out of process with a hard timeout |
| `write_file` | `path`, `content` | inside `owned_paths`; replaces the whole file |
| `delete_file` | `path` | inside `owned_paths`; files only, never directories |
| `run_command` | `command_id`, optional `paths` | the id must be in the allowlist |
| `finish` | the result envelope | ends the packet |

`run_command` carries an **id**, never a command string. Extra `paths` are
validated as workspace-relative paths and capped per command. There is no
action that accepts a shell command, and none that spawns another worker.

## Result envelope

What the worker reports:

```json
{
  "summary": "what you did, 1-3 sentences",
  "files_changed": ["src/parser/core.py"],
  "commands_run": ["pytest"],
  "tests_run": ["pytest"],
  "tests_passed": true,
  "unresolved": ["the streaming path is still untested"],
  "risks": ["callers passing bytes now raise instead of coercing"],
  "recommended_next_step": "run the integration suite against the API server",
  "status": "completed"
}
```

What the **controller** adds, from its own observation:

| Field | Source |
|---|---|
| `observed_files_changed` | `git status` in the worktree, not the worker's claim |
| `command_results` | exit codes the controller recorded |
| `attestation` | provider, reported model, canonical model, usage, cost |
| `denied_actions` | every refused action, with the reason |
| `turns_used`, `attempts`, `duration_s` | measured |
| `claims_contradicted` | true when `files_changed` and `observed_files_changed` disagree |

A writing worker that reports `completed` while the controller observed no
changes is recorded as **failed** with code `no_changes`. Optimistic reports do
not survive contact with the evidence.

## Writing acceptance criteria

| Weak | Strong |
|---|---|
| "the parser works" | "`pytest tests/parser` exits 0" |
| "clean code" | "no function in `src/parser/` exceeds 40 lines" |
| "handles errors" | "malformed input raises `ParseError`, covered by a test" |
| "documented" | "every public function in `src/parser/core.py` has a docstring" |
