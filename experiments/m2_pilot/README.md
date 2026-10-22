# M2 pilot: feasibility of the first two M2 dev tasks

The first real-model runs on the new task format (hidden tests, wrong
candidate fixes, behaviour-only specifications), on 2026-10-22. **This is a
feasibility check on 2 dev tasks × 3 repetitions in one configuration. It is
not a comparison and not a benchmark.** No configuration was tuned.

## Setup

| | |
|---|---|
| Code | commit `df5e503`, tracked files clean; the two pilot scripts were untracked at run time, and their SHA-256 hashes are recorded in `results/manifest.json` |
| Model | `claude-haiku-5-5`; `max_tokens=4096`, `effort=medium`, adaptive thinking with `display=summarized`, parallel tool use disabled, no sampling parameters |
| Harness | plain loop, `max_steps=12`, 1 tool call per step, tools `read_file` / `write_file` / `run_tests` (+ `finish`) |
| Tasks | `feed_pagination_006` (pagination with shared timestamps, 2 files to fix) and `config_merge_007` (configuration merge aliasing). Both are synthetic adaptations of documented bug mechanisms, cited in each `task.toml`. The fixtures were drafted with LLM assistance, then reviewed and validated. |
| Schedule | 3 repetitions, round-robin over tasks (6 runs). Before the first run: the validity checks passed (the original fails, the reference passes public and hidden tests, every wrong candidate fix fails) and a task-set manifest was frozen. It still matched at the end. |
| Scoring | Independent verifier: public tests and hidden tests, each in a fresh Docker sandbox; success = both pass |

Reproduce: `venv/bin/python experiments/m2_pilot/run_pilot.py` (paid; budget guard $0.40), then `analyze.py`.

## Results

| | |
|---|---|
| Verified success | **6/6** (3/3 per task), public and hidden tests passed in every run |
| First tested candidate | passed in **6/6**: the first state the agent ran the tests on was already correct |
| Test feedback | `run_tests` called once per run, always after the fix; **0 failing results seen; 0 edits after a failure** |
| Passed public, failed hidden | 0/6 |
| Stop reason | `finished` in all 6; no run hit the budget |
| Model calls per run | 9–11 (feed_pagination_006: 9, 9, 11; config_merge_007: 10, 10, 10); max 12 allowed |
| Tool errors | 1 (`tool_refused`): one run tried to add a regression test to `tests/test_export.py`; the write allowlist refused it and the agent continued without it |
| Estimated cost | **$0.0208 total**, $0.0026–$0.0053 per run (API-reported tokens, dated price table) |

Per-run details: `results/analysis.md`. Per-run files: `results/<run_id>/`.
Hidden-test output is kept in gitignored `private/` folders and is not published.

**Behaviour was almost identical across repetitions.** Every run read the
README and all source files first, then wrote the fix, ran the tests once
(they passed) and finished. In `feed_pagination_006` all three runs changed
`cursor.py` and `store.py` to use a `(created_at, id)` position, as in the
reference. In `config_merge_007` all three runs rewrote `merge.py` to
deep-copy and return a new dict, with the loader calling it on the defaults
directly. That is a correct fix that differs from the reference (which
deep-copies in the loader); it was one of the two alternative fixes checked
for fairness when the task was built.

**Diff review (all 6, line by line):** every fix is a genuine repair of the
actual mechanism, with no special-casing of test inputs. No run produced one
of the known wrong candidate fixes.

## Interpretation and limitations

- **Ceiling again.** Both tasks were solved on the first tested candidate
  in every run, so test feedback never had a chance to change a decision.
  On these two tasks, a `run_tests` on/off comparison could at most show
  differences in calls, tokens and latency, not in success.
- **What did change compared with P4:** runs used 9–11 of 12 model calls
  (P4: 5–6), because the agent read more files. The budget is now close to
  binding, which matters for any comparison that changes the available tools.
- **Not exercised:** recovery after a failing test, budget exhaustion, and
  the wrong candidate fixes. The hidden tests caught nothing, because nothing
  was wrong.
- **Small n, clustered.** 3 repetitions of 2 tasks are about 2 independent
  observations. Nothing here supports a general difficulty claim.
- **Familiarity.** Both mechanisms (keyset pagination ties, shallow-copy
  aliasing) are well documented, and the fixtures were drafted with an LLM;
  either may make them unusually easy for this model.
- Reasoning summaries are provider-generated and are used here only to form
  hypotheses, not as evidence of why the model acted.
