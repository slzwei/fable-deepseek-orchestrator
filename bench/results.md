# Orchestration benchmark (fabds 0.1.0)

Same task, same repository, same worker action loop, same limits.
Expensive-model stand-in: `claude-opus-5`.

| | A: one expensive agent | B: expensive every step | C: architecture + cheap swarm |
|---|---|---|---|
| planner / architecture calls | 0 | 3 | 1 |
| worker model calls | 9 | 16 | 18 |
| worker model | `claude-opus-5` | `claude-opus-5` | `deepseek-flash` |
| wall clock (s) | 63.7 | 87.8 | 85.7 |
| prompt chars sent | 97,596 | 155,182 | 148,488 |
| packets completed | 1/1 | 3/3 | 3/3 |
| files touched by >1 worker | 0 | 0 | 0 |
| expensive cost reported (USD) | 0.5861 | 1.1013 | 0.0311 |
| integrated suite passes | True | True | True |
| **task actually completed** | yes | yes | yes |

## Notes

- **A**: planner role filled by claude-opus-5, not Fable 5.1 (quota exhausted at benchmark time)
- **B**: planner role filled by claude-opus-5, not Fable 5.1 (quota exhausted at benchmark time)
- **C**: planner role filled by claude-opus-5, not Fable 5.1 (quota exhausted at benchmark time)
- "integrated suite passes" is necessary but not sufficient: a run that changes nothing also leaves the original suite green. "task actually completed" is the real measure - it imports the CLI that was broken at baseline and requires it to work.
- One sample per arm. Indicative, not statistics.
- Cost is what the provider reported. DeepSeek does not report a price, so its usage appears as token counts in the JSON rather than as dollars.
