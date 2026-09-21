# Security boundaries

Three principals, decreasing authority. Each row is enforced in code, not by
asking a model to behave.

| | Controller (you) | Planner (Fable) | Worker (DeepSeek) |
|---|---|---|---|
| Read the repository | yes | no, only the packet it is sent | inside its grant only |
| Write files | yes | **no** | inside `owned_paths`, inside its worktree |
| Run commands | yes | **no** | allowlisted ids only |
| Network | yes | the model call itself | none |
| MCP servers | yours | **none** unless you grant one | none; no MCP surface exists |
| Create workers | yes | no | no |
| Integrate work | yes | no | no |

## What a planner call actually looks like

`--strict-mcp-config` with an empty server set, `--safe-mode`,
`--setting-sources ""`, `--tools ""`, `--permission-prompts none`,
`--no-session-persistence`, a scrubbed environment with no `*KEY*`/`*TOKEN*`
variables and no `CLAUDE_CODE_*` bridge back to the parent session, and an empty
temporary working directory so nothing of yours is auto-discovered.

If you deliberately grant an MCP server, `--safe-mode` is dropped — it disables
MCP wholesale and would silently suppress the grant — and the response records
`mcp_isolated: false` so the reduced isolation stays visible.

`fabds doctor` proves this rather than asserting it: it starts a real fake MCP
server, confirms it initialises under a normal invocation, and confirms it does
**not** under an isolated one.

## What a worker call looks like

Plain HTTPS to the DeepSeek API. No shell, no filesystem, no MCP — there is
nothing to escape from. Every effect on the world goes through the action
protocol, which the controller authorises action by action.

## Always-off

Whatever a packet says:

- `sudo`, `su`, `ssh`, `scp`, `curl`, `wget`, `nc`, `launchctl`, `systemctl`
- `git push`, `git remote`, `git config`, `git clone`, `git filter-branch`
- global package installs (`-g`, `--global`)
- `~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.claude`, `~/.codex`, keychains,
  LaunchAgents, `/etc`, `/usr/bin`
- reading `.env`, `*.pem`, `*.key`, `id_rsa*`, `.netrc`, `.npmrc`,
  `.git-credentials`, cloud credential files — even inside an owned glob

## Trust boundary for repository content

Repository files are data. Both prompts state this explicitly, the content is
wrapped in a delimiter that restates it after the content, and — the part that
matters — the action layer refuses the escalation anyway. A worker that fully
obeys an injected instruction still cannot read outside its grant, write outside
its ownership, or run anything that is not on its allowlist.

## What this does not protect against

- a malicious *controller*: command allowlists are trusted controller input, and
  a controller that authorises `python3 -c <anything>` has authorised anything
- the providers themselves: prompts go to Anthropic and DeepSeek, so do not
  orchestrate over a repository you would not send to either
- a compromised `claude` binary or Python interpreter
- resource exhaustion by a validation command you authorised (it has a timeout,
  but it runs)

See `SECURITY.md` for the full threat model.
