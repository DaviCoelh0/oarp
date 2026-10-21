# Pilot B: hosted-model feasibility (P1–P3)

An informal check, run on 2026-10-21, that Claude Haiku 5.5 can drive oarp-style
native tool calling and recover from tool errors. **This is a feasibility check,
not a benchmark or a harness comparison.** The tasks are trivial arithmetic,
n is tiny, and nothing here says anything about coding ability.

## Setup

| | |
|---|---|
| Model | `claude-haiku-5-5` (API reports `claude-haiku-5-5`) |
| Settings | `max_tokens=4096`, `output_config.effort="medium"`, thinking left at its default (adaptive), no `temperature` (Haiku 5.5 rejects non-default values) |
| Tool choice | `{"type": "auto", "disable_parallel_tool_use": true}` |
| Tools | `add(a: int, b: int)` and `finish(answer: str)`, defined as oarp `ToolSpec`s; `add` runs through the real `oarp.core.tools.ToolRegistry` |
| Loop | Minimal loop written directly against the `anthropic` SDK (1.17.0), not `oarp.core.loop`. `response.content` is appended unchanged so thinking blocks stay valid |
| Budget | Client-side worst-case guard (cap $0.50 for the main run). It is not a hard ceiling: input size is estimated before the call and the real cost is only known afterwards |

The script is `probe_p1_p3.py`. `--dry-run` uses a fake client and makes no API calls.

## Probes

- **P1**: one plain message ("Reply with exactly the word: pong").
- **P2**: 5 arithmetic prompts × 2 repeats, at most 6 model calls each. Integer sums, three terms, a negative number, numbers written as words, and one float case outside the integer-only schema.
- **P3**: tool-error scenarios × 2 repeats, at most 8 model calls each:
  - `transient`: the first `add` call fails, later calls succeed;
  - `down`: `add` always fails;
  - `range` (v1): every argument limited to ±100. **Design flaw:** 150 + 275 cannot be solved under this limit, because every chained sum is above 100 and can't be passed back in;
  - `range_b` (v2, added after the flaw was found): only `b` is limited to ±100, so the task can be solved (150 → 250 → 350 → 425).

## Results

| Probe | Runs | Outcome |
|---|---|---|
| P1 | 1 | Passed ("pong"), 1.1 s |
| P2 integer cases | 8 | 8/8 correct. 0 malformed tool calls. The three-term sum was done as two sequential `add` calls |
| P2 float case | 2 | Both worked around the integer schema (`add(25, 25)`, then divided by 10 → 5) |
| P3 `transient` | 2 | 2/2 retried the identical call once, then finished with 42 |
| P3 `down` | 2 | Retried 2 and 3 times, then reported the failure. Never stated a made-up result. 1 run ended via `finish`, 1 with plain text (`no_tool_call`) |
| P3 `range` (v1, unsolvable) | 2 | Both correctly explained that the task was impossible. 1 via `finish`, 1 with plain text |
| P3 `range_b` (v2) | 2 | 2/2 recovered from the range error by splitting `b` and reached 425 through tool outputs |

**Totals for all real calls: 50 model calls, about $0.0066.** Cost is computed from the `usage` the API returned, at $0.10 / $0.50 per MTok. Per call: mean 594 input / 145 output tokens (main run), median latency 1.0 s, max 8.8 s (the impossible `range` case). 0 truncations, 0 API errors, 0 refusals. 11 of the 40 main-run responses contained (omitted-text) thinking blocks, and sending them back caused no errors.

## Lessons for the harness

1. **Thinking blocks must be stored and sent back unchanged.** oarp's `Message` type can't hold them, so the Anthropic adapter (Component C) has to keep the provider's raw content blocks.
2. **`no_tool_call` is not always a protocol failure.** In 2 runs the model gave up in plain text instead of calling `finish`. The system prompt should say that `finish` ends every run, including failed ones.
3. **Identical runs differ** (2 vs. 3 retries in `down`). Any real evaluation needs repeated runs.
4. **Text-matching graders are unreliable.** The first heuristic marked `range#0` as correct because the model's explanation mentioned 425, which it had computed itself. The runs file for that main run keeps that wrong grade. The `correct_via_tool` field, added later, also requires the value to come from a successful tool call. P4 must use an independent verifier, not answer text.

## Not tested

- How the model handles its *own* malformed tool calls. None occurred, so oarp's argument validation was never triggered by a real model.
- `strict: true` tool schemas, parallel tool calls (disabled here), other effort levels, Sonnet.
- Anything about coding or the oarp loop itself.

## Files

`results/<timestamp>_calls.jsonl` holds one line per API call: settings, request id, stop reason, latency, usage, block types, tool calls, text excerpt. `results/<timestamp>_runs.jsonl` holds one line per run.

| Timestamp | Contents |
|---|---|
| `20261021T040605` | P1 |
| `20261021T040612` | P2 + P3 v1 (`transient`, `range`, `down`) |
| `20261021T044212` | P3 `range_b` |

No credentials appear in any file.
