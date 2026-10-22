# oarp: Observable Agentic RAG Platform

oarp is a Python platform for **building, evaluating and (eventually)
optimizing tool-using LLM agents**. Every run is bounded, traced and scored by
an independent verifier, so "the agent did X" is a claim you can check.

The first task environment is **sandboxed Python bug-fixing**: the agent reads
files, writes fixes and runs tests inside a restricted Docker container, and a
separate verifier decides whether the bug is fixed. Retrieval (RAG)
environments are planned on the same core.

| Status | |
|---|---|
| **Implemented and tested** | provider-neutral agent loop, Anthropic adapter, coding environment, Docker sandbox, independent verifier, JSONL tracing, `oarp` CLI, optional LangGraph version of the loop (215 tests) |
| **Measured with a real model** | a tool-calling pilot and a 15-run coding feasibility study on Claude Haiku 5.5, both small and preliminary ([results](#results-so-far)) |
| **Not yet done** | controlled harness comparisons, held-out evaluation, automatic harness optimization, RAG |

## How a run works

```text
task folder ──► workspace (temp copy) ──► agent loop ◄──► model (Claude Haiku 5.5 / scripted)
                                              │
                                   read_file / write_file      (host, path allowlist)
                                   run_tests ──────────────►   Docker sandbox (no network, read-only)
                                              │
                         final editable files only
                                              ▼
                   independent verifier: fresh copy + pristine tests ──► passed / failed / invalid
                                              │
                   runs/<run_id>/  trace.jsonl · summary.json · final.diff · final/
```

1. **Bounded loop.** `max_steps` limits *model calls*; `max_tool_calls_per_step`
   limits tool calls per response (default 1, so steps and actions line up).
   A run ends with one of four stop reasons: `finished`, `no_tool_call`,
   `max_steps`, `model_error`.
2. **Termination is not success.** The stop reason only says how the run
   ended. Success comes from the verifier alone: a `max_steps` run can have
   fixed the bug, and a `finished` run can be wrong.
3. **Errors are classified where they happen.** Model mistakes (unknown tool,
   bad arguments, a refused write, over the per-step limit) go back to the
   model as error results. A crash in our own tool code is flagged as
   `tool_bug`. Infrastructure failures (e.g. Docker down) abort the run and
   mark it **invalid**, never "failed".
4. **Two implementations of one loop.** `run_agent` is a plain Python loop;
   `run_graph_agent` expresses the same control flow as a LangGraph
   `StateGraph` (a `model` node, a `tools` node, one router). Both use the
   same step functions, model client and tool registry, and equivalence tests
   check that scripted episodes give identical messages, stop reasons, error
   kinds and trace events. `--loop graph` selects it; the choice is recorded
   in the run config and its hash. The plain loop is the dependency-free
   reference and default, and the baseline for all results so far. The graph
   loop is a faithful port, and the place for future structural variants
   (e.g. an added node), which would be compared against that baseline.

### Code layout

| Path | Contents |
|---|---|
| `src/oarp/core/` | Shared and environment-independent: types, `ModelClient` protocol + scripted model, tool registry, loop (plain and optional LangGraph), tracing, pricing |
| `src/oarp/providers/` | `AnthropicModel`: the only module that knows Anthropic's API format |
| `src/oarp/envs/coding/` | Coding environment: task loader, workspace and allowlist, sandbox, tools, verifier, runner |
| `src/oarp/cli.py` | `oarp run` / `oarp rescore` |
| `tasks/coding/dev/` | 5 development tasks, each with a reference solution the agent never sees |
| `docker/` | Sandbox image: base image pinned by digest, packages pinned by version and hash |
| `experiments/` | Studies with their scripts, raw results and READMEs |

A RAG environment would add `envs/rag/` with its own tools and verifier,
without changing `core/`.

## Quick start

Requirements: Python 3.11+, Docker (for running tests and verification), and
an Anthropic API key only for paid runs. Every command below was run on
2026-10-22 (macOS, Python 3.14, Docker 29.6).

```bash
python3 -m venv venv
venv/bin/python -m pip install -e '.[dev]'
venv/bin/python -m pip install -e '.[langgraph]'  # optional: the graph loop (--loop graph)
docker build -t oarp-sandbox:0.1 -f docker/sandbox.Dockerfile docker
```

```bash
venv/bin/python -m pytest -q                      # 215 passed, 1 skipped (opt-in live test)
                                                  # without the langgraph extra: 181 passed, 4 skipped
venv/bin/python -m pytest -q -m "not docker"      # without Docker: 183 passed, 1 skipped
```

Free, scripted dry run through the full pipeline (no API calls; the scripted
model changes nothing, so the task is correctly scored as failed):

```bash
venv/bin/oarp run --task tasks/coding/dev/off_by_one_001 --dry-run
venv/bin/oarp run --task tasks/coding/dev/off_by_one_001 --dry-run --loop graph  # same run, graph loop
```

**Paid** run with Claude Haiku 5.5. It needs `ANTHROPIC_API_KEY=...` in a
gitignored `.env` file and costs about $0.001 per task. The printed cost is a
client-side *estimate*, so set a spend limit in your Anthropic Console.

```bash
venv/bin/oarp run --task tasks/coding/dev/off_by_one_001 --runs-dir runs
venv/bin/oarp rescore runs/<run_id>               # re-verify the saved files, compare hashes
```

`oarp run` exits with 0 (verified success), 1 (verified failure) or 3
(invalid run). `oarp rescore` exits with 0 when status and file hashes match
the recorded run, and 1 when they don't.

## Results so far

All results are small and preliminary. Each study's README has the full setup
and limitations.

**Tool-calling pilot** ([experiments/pilot_b](experiments/pilot_b/README.md)).
50 Claude Haiku 5.5 calls on arithmetic tasks with deliberately failing tools,
about $0.007 in total. Tool calls were always well-formed, transient errors were
retried, and an always-failing tool was reported honestly, never faked. It
showed two things: provider thinking blocks must be replayed unchanged, and
answer-text graders are unreliable. This is not a coding benchmark.

**Coding feasibility, P4** ([experiments/p4_feasibility](experiments/p4_feasibility/README.md)).
5 dev tasks × 3 repetitions at commit `9965ca0`, one harness configuration
(`max_steps=12`, 1 tool call per step, `run_tests` available).

| | |
|---|---|
| Verified success | **15/15**, 3/3 on every task |
| Model calls per run | 5–6 of 12 allowed; 0 tool errors or rejected calls |
| Estimated cost | $0.0173 in total (about $0.001 per run) |
| Gaming check | all 15 diffs reviewed by hand: genuine, minimal fixes |

What this shows: **the pipeline works end to end with a real model.** What it
does not show: anything about harder tasks or about harness choices. It is a
**ceiling effect**. The tasks are small, their docstrings state the intended
behaviour, and the agent always fixed the bug *before* running the tests, so
test feedback never influenced a decision. The reported Wilson 95% interval
(0.80–1.00) treats the 15 runs as independent. Three repetitions of one task
are not independent, though: the effective sample is closer to 5 tasks than
to 15 runs.

**CLI run** ([experiments/m1_cli_run](experiments/m1_cli_run/README.md)). One
`oarp run` on `off_by_one_001` at commit `a38ce5e`: verified success in 5 model
calls, $0.001, and `oarp rescore` reproduced it with matching file hashes.

## Security model

Model inference and the API key stay in the trusted host process. Only
`run_tests` and the verifier execute code, each time in a fresh container:

- `--network none`, non-root user (65534), all Linux capabilities dropped, `no-new-privileges`
- read-only root filesystem and read-only mounts; only a 64 MB in-memory `/tmp` is writable
- 1 CPU, 512 MB memory (no swap), 128 processes, per-task timeout; containers are removed by name on timeout
- no host environment variables are passed in, no Docker socket, never `--privileged`
- the original tests are mounted over the agent's copy, and the agent's tools can only write the task's listed editable files (an allowlist, so e.g. a new `conftest.py` is rejected)

Unit tests check that every flag is passed. Integration tests run real
containers to check the network, user, read-only filesystem, absence of host
secrets, timeout cleanup, and process and memory limits. A control run without
the restrictions confirmed that the network, user and process tests detect
their absence. The CPU and `/tmp` size limits are set but not tested.
**Limitations:** containers share the Docker Desktop VM's
kernel, so this protects against accidental damage and score-gaming by buggy
code, not against a determined attacker. It also cannot stop a "fix" that
special-cases the visible test inputs: hidden tests are planned, and passing
diffs are reviewed by hand until then.

**Reproducibility.** Every trace records the git commit (and whether tracked
files were modified), hashes of the prompts, tools and configuration, the
model ID the API reported, SDK and Python versions, and the sandbox image ID.
The image's inputs are pinned, but Docker builds are not bit-for-bit
reproducible: a rebuild gets a new image ID with the same package versions.
The committed P4 and CLI runs still rescore identically on a rebuilt image.
Model outputs are stochastic (Claude Haiku 5.5 does not accept a non-default
temperature), so results are reported over repeated runs.

## What's next

1. **Informative tasks:** less revealing specifications, multi-file bugs, and tasks where a plausible first fix fails, so test feedback can matter. A separate held-out set is never used for tuning.
2. **One controlled comparison:** fixed model, settings, tasks and verifier, changing a single harness variable (e.g. `run_tests` available or not), with repeated runs and uncertainty reported. Better metrics too: tests *run* vs. failing results *seen* vs. an edit made *after* a failure.
3. **Harness optimization:** a bounded search over harness configurations on dev tasks, validated on held-out tasks.
4. **RAG environment:** retrieval tools and retrieval-specific evaluation on the same core.

oarp does not yet demonstrate automatic self-improvement or general coding
ability. Those are research directions, not results.
