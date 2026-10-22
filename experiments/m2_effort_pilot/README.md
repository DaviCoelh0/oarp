# M2 effort pilot: low vs medium effort on two dev tasks

A model-configuration feasibility study on 2026-10-22. The only variable was
the model's `effort` setting (`low` vs the default `medium`). **This is not a
harness comparison and not evidence of harness optimization.** It asks
whether a cheaper setting takes these tasks off the ceiling seen in the
[M2 pilot](../m2_pilot/README.md).

## Setup

| | |
|---|---|
| Code | commit `217ccbf`, clean working tree, no untracked files (recorded in `results/manifest.json`, with the scripts' SHA-256) |
| Model | `claude-haiku-5-5`; `max_tokens=4096`, adaptive thinking with `display=summarized`, parallel tool use disabled, no sampling parameters; **`effort` = `medium` or `low`** |
| Harness | plain loop, `max_steps=12`, 1 tool call per step, `read_file` / `write_file` / `run_tests` (+ `finish`), identical prompts |
| Tasks | `feed_pagination_006`, `config_merge_007` (synthetic adaptations of documented bug mechanisms; fixtures drafted with LLM assistance, then reviewed and validated) |
| Schedule | 3 repetitions; each repetition runs all four (task, effort) conditions in turn (12 runs). Before the first run: validity checks passed and a task-set manifest was frozen, and it still matched at the end. |
| Scoring | Independent verifier: public and hidden tests, each in a fresh Docker sandbox; success = both pass |

Reproduce: `venv/bin/python experiments/m2_effort_pilot/run_effort_pilot.py` (paid; budget guard $0.20), then `analyze.py`.

## Results

| Task | Effort | Verified success | First tested candidate correct | Failing results seen | Median model calls | Median in/out tokens | Median cost | Median wall s |
|---|---|---|---|---|---|---|---|---|
| feed_pagination_006 | medium | 3/3 | 3/3 | 0/3 | 9 | 25,236 / 2,145 | $0.0036 | 19.5 |
| feed_pagination_006 | low | 3/3 | 3/3 | 0/3 | 8 | 19,253 / 1,846 | $0.0028 | 17.0 |
| config_merge_007 | medium | 3/3 | 3/3 | 0/3 | 10 | 19,477 / 1,405 | $0.0027 | 17.2 |
| config_merge_007 | low | 3/3 | 3/3 | 0/3 | 9 | 17,479 / 1,247 | $0.0024 | 15.1 |

- **12/12 verified successes**, public and hidden tests passed in every run;
  no run passed public but failed hidden; no run hit the budget; no tool errors.
- **Test feedback was never needed:** every run called `run_tests` exactly
  once, after its fix, and it passed. 0 failing results seen, 0 edits after
  a failure.
- **Estimated cost: $0.0361 total.** The 6 `low` runs cost $0.0168 together
  and the 6 `medium` runs $0.0193 (about 13% less for `low`).
- **What `low` changed:** about one model call fewer per run, because it read
  fewer files. It skipped `tests/test_export.py` in 2 of 3 pagination runs and
  `__init__.py` in all three merge runs. It did not skip the final test run.

Per-run details and tool sequences: `results/analysis.md`. Hidden-test output
is kept in gitignored `private/` folders.

**Diff review (all 12, line by line):** every fix is a genuine repair of the
actual mechanism, with no special-casing of test inputs; no known wrong fix
appeared. For `config_merge_007`, runs at both effort levels split between
copying in the loader (the reference approach) and a deep-copying `merge`
(an alternative correct fix validated when the task was built).

## Interpretation and limitations

- **The ceiling persists at `low` effort.** On these two tasks, lowering
  effort reduced tokens and calls but did not reduce verified success, so
  it created no failures for a harness change to recover.
- **No equivalence claim.** 3 runs per condition on 2 tasks (about 2
  independent observations) cannot show that `low` is as reliable as
  `medium` in general, only that no difference appeared here.
- **The cost difference is small and descriptive.** It comes mostly from
  reading fewer files; no uncertainty estimate is given at this n.
- **Not exercised:** recovery after a failing test, budget exhaustion, the
  known wrong fixes, and hidden-test catches.
- Both mechanisms are well documented and the fixtures were drafted with an
  LLM, which may make them unusually easy for this model.
