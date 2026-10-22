# First run through the `oarp` CLI

One paid run, on 2026-10-22, recorded to show the command-line path end to
end. It is a demonstration, not a measurement: n = 1 on an easy dev task that
P4 had already solved 3/3.

```bash
venv/bin/oarp run --task tasks/coding/dev/off_by_one_001 --runs-dir experiments/m1_cli_run
venv/bin/oarp rescore experiments/m1_cli_run/20261021T232258-871ca4
```

| | |
|---|---|
| Code | commit `a38ce5e`, no modified tracked files (`untracked_files: 1` is this run's own output folder) |
| Model / harness | `claude-haiku-5-5` with the P4 settings; `max_steps=12`, 1 tool call per step, `run_tests` available |
| Outcome | verified success (`passed`, 5/5 tests), stop reason `finished`, exit code 0 |
| Behaviour | read `src/stats.py` → read the tests → write the fix → `run_tests` → `finish` (5 model calls, 4 tool calls) |
| Tokens / cost | 6,570 in / 682 out; estimated $0.000998 |
| Rescore | `now passed, recorded passed; file hashes match` (exit code 0) |

The fix is `range(len(values) - window + 1)`; see `final.diff`. One of the 5
model calls (the write) carried a provider reasoning summary.
