# When a second Fable call is worth paying for

The default answer is **no**. A run uses at most `max_planner_rounds` planner
calls (default 2), and the second one happens only when a named trigger fires.
This is what stops a planner/worker ping-pong loop from forming.

## Triggers

A second call happens when any of these is true:

| Trigger | Why it is worth the money |
|---|---|
| two or more attempted fixes failed | the approach is probably wrong, not the execution |
| the architecture changed materially mid-run | the original plan was for a different system |
| half or more of the packets failed or were blocked | the decomposition did not survive contact |
| a worker's report contradicts observed changes | something is wrong that a worker cannot see |
| a validation command reported failing tests | a design flaw may be showing through |
| two or more workers hit permission boundaries | the ownership map is probably wrong |
| a closing review was explicitly requested | you decided the stakes justify it |

## Non-triggers

None of these earns a Fable call. Handle them yourself:

- one worker failed and a retry fixed it
- a worker asked a question the packet already answered
- tests pass and the diff looks reasonable
- you want reassurance
- you want the plan restated

## What a critique call gets

Not the repository. A compact record: the plan, per-packet status, observed
changes, the controller's own validation results, and the specific question that
triggered the escalation. The critic has no tools and cannot verify anything, so
it is asked to separate "the evidence shows this is broken" from "the evidence
does not cover this".

## What to do with the verdict

| Verdict | Action |
|---|---|
| `sound` | integrate, verify the merged result, finish |
| `needs_work` | fix the specific findings; do not re-plan the whole task |
| `unsound` | stop delegating, reconsider the approach, possibly do it yourself |

A finding marked `unverified` is a hypothesis. Check it before acting on it —
the critic could not run anything.
