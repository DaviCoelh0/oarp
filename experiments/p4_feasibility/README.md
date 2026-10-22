# P4: preliminary coding-agent feasibility study

The first runs of the full oarp pipeline with a real model, on 2026-10-21/22:
Claude Haiku 5.5 → bounded tool loop → sandboxed tests → independent verifier
→ JSONL traces. **This is a feasibility check on 5 easy dev tasks, not a
benchmark and not a harness comparison.** No configuration was tuned.

## Setup

| | |
|---|---|
| Code | commit `9965ca0`, clean working tree (recorded in every trace) |
| Model | `claude-haiku-5-5`; `max_tokens=4096`, `effort=medium`, adaptive thinking with `display=summarized`, parallel tool use disabled, no sampling parameters |
| Harness | `max_steps=12`, 1 tool call per step, tools `read_file` / `write_file` / `run_tests` (+ `finish`) |
| Tasks | 5 dev tasks in `tasks/coding/dev/`: `off_by_one_001`, `mutable_default_002`, `merge_intervals_003`, `cart_rounding_004` (bug in a helper module, 2 editable files), `lru_cache_005` |
| Schedule | 3 repetitions, round-robin over tasks (15 runs) |
| Scoring | Independent verifier: original task + final editable files only + pristine tests, in the Docker sandbox; success = pytest exit 0 |

Reproduce: `venv/bin/python experiments/p4_feasibility/run_p4.py` (paid; budget guard $1.00), then `analyze.py`.

## Results

| | |
|---|---|
| Verified success | **15/15** (Wilson 95% interval 0.80–1.00); 3/3 on every task |
| Stop reason | `finished` in all 15 runs; no run reached the step budget |
| Model calls per run | 5 (6 for `cart_rounding_004`); max 12 allowed |
| Tool errors / rejected calls | 0 / 0 (so error recovery was never exercised) |
| Estimated cost | **$0.0173 total**, about $0.001 per run (from API-reported tokens and the dated price table) |
| Latency | median 1.1 s per model call; median 6.6–9.0 s per run |
| Reasoning summaries | 19 of 78 model calls; replaying summarized thinking blocks caused no API errors |

The full table is in `results/analysis.md`; per-run files are in `results/<run_id>/`.

**Behaviour was nearly deterministic.** All 3 repetitions of each task used
the identical tool sequence: read the source (and, for the cart task, the
helper too) → read the tests → write the fix → `run_tests` once → `finish`. The
agent never ran the tests before fixing. The single test run confirmed a fix
that already passed, so test feedback never changed a decision.

**Gaming check (all 15 diffs reviewed line by line):** every fix is a genuine,
minimal repair of the actual bug, with no special-casing of test inputs. In
`cart_rounding_004` all 3 fixes were made in the helper (`money.py`), where
the bug is. 13 of 15 diffs are identical, or nearly so, to the reference
solutions.

## Interpretation and limitations

- **Ceiling effect.** These tasks are too easy for Haiku 5.5 at this budget.
  A harness comparison (e.g. with vs. without `run_tests`, or a different
  step budget) on these tasks could not show a difference, because nothing
  can improve on 15/15.
- **The tasks give the bug away.** Each docstring states the intended
  behaviour precisely ("halves rounded up", "touching intervals merge",
  "`get()` counts as a use"). The model's visible text and reasoning
  summaries show it found the bug by comparing code with docstring, before
  reading the tests.
- **Task familiarity.** The tasks are textbook bug patterns written for this
  study. Patterns this common are probably familiar to an LLM from training
  data, which may make them unusually easy.
- **Small n, one model, one configuration.** The intervals are wide, and the
  absence of failures here says little about harder tasks.
- **The interval is optimistic.** The Wilson interval (0.80–1.00) treats the
  15 runs as independent trials. Repetitions of the same task share that
  task's difficulty (they are clustered), so the effective sample size is
  closer to 5 tasks than to 15 runs. *(Added 2026-10-22 after review.)*
- **`used_test_feedback` means "`run_tests` was called".** It does not mean a
  failing result changed the fix. Here every call happened after a correct
  fix. Future studies should count tests run, failing results seen, and
  edits made after a failure separately. *(Added 2026-10-22.)*
- **Sandbox image.** All runs used image `sha256:5637afa0…`, which is no
  longer present locally. A rebuild from the same pinned inputs gets a new
  image ID with identical package versions; the runs rescore identically on
  it (`oarp rescore`). *(Added 2026-10-22.)*
- **Not exercised:** tool-error recovery, budget exhaustion, test-feedback
  dependence, gaming attempts, multi-attempt fixes.
- Reasoning summaries are provider-generated summaries, used here only to
  form hypotheses, not as evidence of why the model acted.
