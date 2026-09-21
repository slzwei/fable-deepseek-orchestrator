# Benchmark

```bash
python3 bench/benchmark.py --repo <baseline-repo> --workdir /tmp/fabds-bench \
    --arms A,B,C --expensive-model claude-opus-5 --out bench/results.json
```

## What is being compared

Three architectures, same task, same repository, same worker action loop, same
limits. The only variables are how many expensive calls happen and who does the
implementation.

| Arm | Architecture |
|---|---|
| **A** | one expensive generalist does everything, no delegation |
| **B** | the expensive model is consulted before every packet *and* implements them |
| **C** | one architecture call, then a swarm of cheap workers, controller verifies |

The task: implement `readability_score()` to a written spec in a small Python
package, plus tests. Baseline state is genuinely broken — the package's CLI
raises `ImportError` — so "did it work" is an objective question, not a matter
of taste.

**Completion is measured by importing and running the CLI that was broken at
baseline**, not by the test suite going green. A run that changes nothing also
leaves the original suite green, so the suite alone cannot distinguish success
from doing nothing.

## Results

fabds 0.1.0, 2026-09-22, macOS, one sample per arm.

| | A: one expensive agent | B: expensive every step | C: architecture + cheap swarm |
|---|---|---|---|
| planner / architecture calls | 0 | 3 | 1 |
| worker model calls | 9 | 16 | 18 |
| worker model | `claude-opus-5` | `claude-opus-5` | `deepseek-flash` |
| wall clock (s) | 63.7 | 87.8 | 85.7 |
| prompt chars sent | 97,596 | 155,182 | 148,488 |
| packets completed | 1/1 | 3/3 | 3/3 |
| files touched by >1 worker | 0 | 0 | 0 |
| expensive cost reported (USD) | 0.5861 | 1.1013 | **0.0311** |
| **task actually completed** | yes | yes | yes |

### Reading this honestly

**C wins decisively on cost and nothing else here.** Expensive-model spend is
**18.8x lower than A** and **35.4x lower than B**, for the same completed task.
That is the result the architecture is designed to produce, and it does.

**A is the fastest, and that is not a flaw in C — it is the point of the skill's
guidance.** This task is small enough for one capable agent to hold entirely in
its head. Orchestration adds a planning round, three worktrees and a
verification pass, and on a task this size that overhead is not repaid in wall
time. `SKILL.md` says not to orchestrate work below roughly twenty minutes of
direct effort, and this benchmark is evidence for that advice rather than
against it. The cost advantage survives anyway.

**B is the worst of both.** It pays the most, is no faster than C, and the extra
planner calls bought nothing: consulting an expensive model before every packet
is a pure tax once the architecture is settled. This is the arrangement fabds's
escalation policy exists to prevent.

**C makes more model calls but sends comparable context.** Eighteen cheap calls
against nine expensive ones: DeepSeek needs more turns per unit of work, which
is exactly the trade being made. Optimising for fewer model calls would select
the most expensive architecture.

**Neither B nor C duplicated work.** Zero files were touched by more than one
worker, because ownership is assigned before dispatch and overlapping claims are
serialised rather than raced.

## The first run, and why it is worth reporting

On the first attempt, **A completed and both B and C failed** — on the same
single assertion. The spec said sentences are terminated by `.`, `!` or `?` but
never said what to do with a trailing fragment carrying no terminator. The
implementing worker counted it as a sentence; the testing worker assumed it was
not one. Both readings were defensible.

Arm A never hit this, because one agent wrote the implementation and the tests
and was internally consistent — it silently picked a reading and agreed with
itself.

Two things follow, and both are real:

1. **The benchmark task was underspecified**, so the first run measured spec
   luck rather than architecture. The README was made exact and all three arms
   were re-run. The numbers above are from that fair run.
2. **Splitting implementation from testing surfaces ambiguity that a single
   agent conceals.** C's read-only audit worker had already reported this exact
   ambiguity, unprompted, in the dogfood run — *"Sentence count undefined when
   text has no terminator"* — before any test failed. A single agent produces a
   consistent artefact built on an unexamined assumption; a swarm produces a
   disagreement that points straight at the underspecified requirement.

Whether that is a cost or a benefit depends on the work. For a throwaway
script, it is friction. For anything with a specification worth honouring, being
told your spec is ambiguous is the more valuable output.

## Limitations

- **One sample per arm.** Model runs vary; treat single-digit percentage
  differences as noise. The cost gap is an order of magnitude and is not noise.
- **The planner role was filled by `claude-opus-5`, not Fable 5.1**, because the
  Fable quota was exhausted when the benchmark ran. That affects the *quality*
  of planning, not the *shape* of the architectures, so call counts, wall time,
  context volume and duplicated work remain comparable. A run with Fable would
  change the expensive-model cost per call in every arm that uses it, in the
  same direction.
- **Cost is what the provider reported.** The Claude CLI reports a dollar
  figure; DeepSeek does not, so its usage appears as token counts in
  `results.json`. No price is invented here. C's true cost is its reported
  figure *plus* an unpriced DeepSeek component, which is why the comparison is
  labelled "expensive cost" rather than "total cost".
- **One task, one repository, one language.** A task with genuinely independent
  subsystems would favour C's parallelism far more than this one does; a
  one-line fix would favour A even more.
