# oarp: Observable Agentic RAG Platform

oarp is a Python platform for **building, evaluating and optimizing
tool-using LLM agents** (optimizing so far means a bounded search over harness
settings with the model frozen). Every run is bounded, traced and scored by
an independent verifier, so "the agent did X" is a claim you can check.

The first task environment is **sandboxed Python bug-fixing**: the agent reads
files, writes fixes and runs tests inside a restricted Docker container, and a
separate verifier decides whether the bug is fixed. A second environment,
**retrieval** over a scientific-claim benchmark (BEIR SciFact), asks when
extra retrieval work is worth its cost: BM25 search, a model reranker, and a
search agent built as a LangGraph state graph whose harness enforces the
search budget. Every comparison there was pre-registered, run on development
queries, and the final ones were evaluated once on the untouched test split.

**Headline (held-out test split, 300 queries, nDCG@10):** BM25 0.662 → model
reranking 0.757 → search agent 0.791. A policy chosen by an automated search
over 1,089 routing configurations reached 0.786 at about half the agent's
cost; its pre-registered non-inferiority to the agent was not established
([details](#held-out-evaluation)).

| Status | |
|---|---|
| **Implemented and tested** | provider-neutral agent loop, Anthropic adapter, coding environment, Docker sandbox, independent verifier, JSONL tracing, `oarp` CLI, optional LangGraph version of the loop, task format with hidden tests and wrong candidate fixes, frozen task-set manifests; retrieval data loader, BM25 search, retrieval metrics, listwise reranker and relevance gate, RM3 query expansion, a LangGraph search agent with one config dataclass and one graph builder, shared study machinery (preflight, budget guards, pinned inputs, stratified bootstrap) (609 tests) |
| **Measured with a real model** | a tool-calling pilot and three small coding feasibility studies on Claude Haiku 5.5 (33 runs, all at a success ceiling); retrieval studies of reranking, gating, a search agent and a fixed reformulation on development queries, and a one-shot evaluation on the held-out test split ([results](#results-so-far)) |
| **Measured without a model** | a BM25 baseline, upper bounds of reranking and gating, reachability and RM3, an offline search-gate study, and an automated search over 1,089 routing policies ([results](#results-so-far)) |
| **Not yet done** | agent-proposed harness changes, a second retrieval benchmark, transfer of the retrieval findings back to the coding environment |

## How a run works

```text
task folder ──► workspace (temp copy) ──► agent loop ◄──► model (Claude Haiku 5.5 / scripted)
                                              │
                                   read_file / write_file      (host, path allowlist)
                                   run_tests ──────────────►   Docker sandbox (no network, read-only)
                                              │
                         final editable files only
                                              ▼
          independent verifier: fresh copy + pristine public tests (+ hidden tests) ──► passed / failed / invalid
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
   reference and default, and the baseline for all coding results. The graph
   loop is a faithful port, and the place for future structural variants
   (e.g. an added node), which would be compared against that baseline.

### Tasks and evaluation levels

A task is a folder: `task.toml`, a starting `repo/` (source plus the
**public tests** the agent can read and run), and a reference `solution/` the
agent never sees. Format 2 adds `hidden_tests/`, which only the verifier runs
(in a second fresh container, on the same final files), and `decoys/`:
deliberately wrong but plausible fixes. Before a task is used, automated
checks require that the original fails its public tests, the reference
passes public and hidden tests, every decoy fails, at least one decoy fails
the public tests, and the starting repo does not contain the reference or a
decoy. Results report public and hidden outcomes separately, so a fix that
only fits the visible tests shows up as a failure.

Three levels are kept apart: public tests (seen by the agent), hidden tests
of development tasks (never seen by the agent, but their scores may guide
task selection and tuning), and **held-out** tasks (`split = "test"`), which
are not loaded or run without an explicit opt-in and are never used for
tuning. Hidden-test output is written only to a run's `private/` folder,
which is not published. `oarp tasks freeze` records a hash of every task file
in a manifest; `oarp tasks check` fails if any task changed since.

### Code layout

| Path | Contents |
|---|---|
| `src/oarp/core/` | Shared and environment-independent: types, `ModelClient` protocol + scripted model, tool registry, loop (plain and optional LangGraph), tracing, pricing |
| `src/oarp/providers/` | `AnthropicModel`: the only module that knows Anthropic's API format |
| `src/oarp/envs/coding/` | Coding environment: task loader, workspace and allowlist, sandbox, tools, verifier, runner, task validity checks, task-set manifests |
| `src/oarp/envs/retrieval/` | Retrieval environment: checksum-verified benchmark download and loader, deterministic BM25 search (also weighted queries), retrieval metrics, model reranker and relevance gate (`rerank.py`, `gate.py`), RM3 expansion (`rm3.py`), the search agent as a LangGraph `StateGraph` (`agent.py`: `AgentConfig`, `build_agent_graph`), label-free strategy pieces (`strategies.py`), study machinery (`study.py`), statistics (`stats.py`), label-reading analysis helpers kept apart (`analysis.py`) |
| `src/oarp/cli.py` | `oarp run` / `oarp rescore` / `oarp tasks freeze\|check` |
| `tasks/coding/dev/` | 7 development tasks, each with a reference solution the agent never sees (2 in format 2, with hidden tests and wrong candidate fixes) |
| `docker/` | Sandbox image: base image pinned by digest, packages pinned by version and hash |
| `experiments/` | Studies with their scripts, raw results and READMEs |

The retrieval environment adds `envs/retrieval/` with its own data, tools and
evaluation, without changing `core/`. Benchmark data is downloaded locally
into a gitignored `data/` folder and is never committed.

## Quick start

Requirements: Python 3.11+, Docker (for running tests and verification), and
an Anthropic API key only for paid runs. The commands below were run on
2026-10-22 (macOS, Python 3.14, Docker 29.6); the test counts were re-run on
2026-10-27 with all extras installed.

```bash
python3 -m venv venv
venv/bin/python -m pip install -e '.[dev]'
venv/bin/python -m pip install -e '.[langgraph]'  # optional: the graph loop and the retrieval agent
venv/bin/python -m pip install -e '.[retrieval]'  # optional: the retrieval environment
docker build -t oarp-sandbox:0.1 -f docker/sandbox.Dockerfile docker
```

```bash
venv/bin/python -m pytest -q                      # 609 passed, 1 skipped (opt-in live test)
venv/bin/python -m pytest -q -m "not docker"      # without Docker: 529 passed, 1 skipped, 80 deselected
# tests that need a missing optional extra (langgraph, retrieval) are skipped, not failed
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
venv/bin/oarp tasks freeze --split dev --output manifest.json   # hash every dev task
venv/bin/oarp tasks check manifest.json           # fails if any task changed since
```

Retrieval, free (no model calls; downloads about 2.8 MB once):

```bash
venv/bin/python -m oarp.envs.retrieval.data                     # download and verify the data
venv/bin/python experiments/retrieval_baseline/run_baseline.py  # BM25 on the development queries
venv/bin/python experiments/retrieval_rerank_headroom/run_analysis.py  # reranking and gating upper bounds
venv/bin/python experiments/retrieval_s3_reachability/run_analysis.py  # reachability and RM3
venv/bin/python experiments/retrieval_s3_pilot/run_pilot.py --dry-run  # the search agent, scripted model
```

The paid retrieval studies each have a runner with a `--dry-run` mode and a
README with the exact command, settings and budget. Their per-query records
stay in gitignored `private/` folders, so the offline analyses that reuse them
(the gate study, the configuration search) need those local records.

`oarp run` exits with 0 (verified success), 1 (verified failure) or 3
(invalid run). `oarp rescore` exits with 0 when status and file hashes match
the recorded run, and 1 when they don't.

## Results so far

Each study's README has the full setup, settings, budget and limitations.
Every retrieval comparison followed a written contract (hypothesis, margin,
budget, outcome rules) fixed before its run; development queries were split
into a tune half (for choices) and a check half (for confirmation), and the
test split was opened once, for the final evaluation below. Most results are
small; negative and inconclusive ones are reported like the others.

### Held-out evaluation

[experiments/retrieval_heldout](experiments/retrieval_heldout/README.md):
the one-shot evaluation on SciFact's 300 test queries, which no choice had
used. Everything was frozen before the test split was loaded. Claude Haiku
5.5, one S1 call and one agent episode per query, $1.76 in total.

| | BM25 (S0) | Reranking (S1) | Search agent (S3) | Searched policy (P\*) |
|---|---|---|---|---|
| **nDCG@10** | 0.662 | 0.757 | **0.791** | 0.786 |
| $ per query (estimated) | 0 | 0.0013 | 0.0042 | 0.0020 |
| s per query | 0 | 4.5 | 11.0 | 5.9 |

| Pre-registered comparison | Difference [95% CI] | Outcome |
|---|---|---|
| S1 − S0 | +0.095 [+0.067, +0.125] | holds |
| S3 − S1 | +0.034 [+0.016, +0.055] | holds |
| P\* − S1 (primary) | +0.030 [+0.014, +0.048] | holds |
| P\* − S3 ≥ −0.01, at lower cost | −0.004 [−0.015, +0.004], at 49% of S3's cost | not shown |

- **S1:** a model reranks BM25's top 20 in one call.
- **S3:** a search agent that may search the corpus again (at most twice)
  before its candidates are reranked.
- **P\*:** the policy an automated configuration search selected on
  development data. It routes each query by BM25's own scores to S3, S1 or
  plain BM25.
- **Each step of retrieval work paid off on untouched queries.** The routed
  policy beats reranking at half the agent's cost.
- **Not shown:** that it matches the agent within the pre-set 0.01 margin.
  The combined optimisation claim (both of the last two rows) is therefore
  **not established**.
- **Where the agent's gain comes from:** almost entirely the 48 test queries
  where BM25 had no relevant document in its top 20 (0.000 → 0.211). On the
  others it neither helped nor harmed (+0.001).
- **Limits:** one run of a stochastic model, one dataset, one model; SciFact
  may be in the model's training data.

### Development studies (in order)

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

**Harder coding tasks** ([experiments/m2_pilot](experiments/m2_pilot/README.md),
[experiments/m2_effort_pilot](experiments/m2_effort_pilot/README.md)). Two
multi-file tasks with behaviour-only specifications, hidden tests and known
wrong fixes, 3 runs each, then the same tasks at the model's `low` and `medium`
effort settings (12 more runs). All 18 runs were verified successes, and in
every run the first state the agent tested was already correct: test feedback
never changed a decision, and no known wrong fix appeared. `low` effort used
about one model call fewer and about 13% less estimated cost. This is the same
ceiling as P4, so these tasks cannot show whether a harness change helps;
this is why work moved to retrieval, where success is graded rather than
pass/fail.

**Retrieval baseline** ([experiments/retrieval_baseline](experiments/retrieval_baseline/README.md)).
Plain BM25 on the 809 development queries of a scientific-claim retrieval
benchmark (5,183 abstracts), no model calls: nDCG@10 0.667, Recall@10 0.787,
Recall@100 0.906. The first relevant document is at rank 1 for 56% of queries,
at ranks 2–100 for 35%, and not in the top 100 for 9%, so better ranking has
more room to help than more searching. The held-out test queries were not used.

**Reranking headroom** ([experiments/retrieval_rerank_headroom](experiments/retrieval_rerank_headroom/README.md)).
Free upper bounds on the same queries. A perfect reranker of BM25's top 20
would reach nDCG@10 0.845. A perfect gate that reranks only queries whose
rank 1 is not relevant (44%) would reach 0.841. A rule gate on BM25's score gap
reaches 0.784 at a similar rerank rate. These numbers use the labels; no
model was called.

**Reranking pilot** ([experiments/retrieval_s1_pilot](experiments/retrieval_s1_pilot/README.md)).
Claude Haiku 5.5 reranked BM25's top 20 in one call per query, on 100
development queries. nDCG@10 rose from 0.698 to 0.754 (+0.056, 95% CI
+0.021 to +0.095) at about $0.0013 and 4–5 s per query, about 40% of what a
perfect reranker of the same candidates could gain. Showing the candidates
shuffled instead of in BM25 order made no clear difference (−0.004, CI −0.013
to +0.002). 2.5% of calls were declined by the provider's safety filter on
biomedical claims. A pilot on development queries, not a held-out result.

**Reranking confirmation** ([experiments/retrieval_s1_check](experiments/retrieval_s1_check/README.md)).
The same reranker, unchanged, on 404 other development queries that no
setting had been chosen on: nDCG@10 rose from 0.666 to 0.786 (+0.120, 95% CI
+0.096 to +0.144), 65% of what a perfect reranker of the same candidates
could gain. Almost all of the gain comes from queries whose first relevant
document BM25 placed at ranks 2–20. Development data, not a held-out result.

**Rerank gate** ([experiments/retrieval_s2b_pilot](experiments/retrieval_s2b_pilot/README.md)).
A cheap model call deciding whether to rerank (is BM25's first result
relevant?) halved the cost of always reranking but lost part of the gain. It
was not reliably better than a free rule on BM25's score gap at the same
rerank rate, which was cheaper and kept more of the gain. A perfect gate
would beat always reranking at 38% of the calls. A pilot on 100 development
queries.

**Score-gap threshold** ([experiments/retrieval_s2_tau](experiments/retrieval_s2_tau/README.md)).
A threshold chosen on 405 development queries to rerank as few queries as
possible within 0.01 nDCG@10 of always reranking. It reranked half of them.
On 404 other development queries it halved cost and latency, but lost 0.026
nDCG@10 (95% CI 0.015 to 0.039) against always reranking, more than the
pre-stated 0.02 margin. Not shown. This confirmation was not blind: the
rule's curve on those queries had been seen before the threshold rule was
fixed.

**Search reachability and RM3** ([experiments/retrieval_s3_reachability](experiments/retrieval_s3_reachability/README.md)).
Free, on 405 development queries. 63 (16%) have no relevant document in
BM25's top 20, which reranking cannot fix. RM3 query expansion (untuned)
lowered nDCG@10 by 0.044 [0.024, 0.065] and brought a relevant document into
the top 20 for only 10 of the 63, so a fixed classic reformulation was a weak
baseline.

**Search agent, pilot** ([experiments/retrieval_s3_pilot](experiments/retrieval_s3_pilot/README.md)).
The agent is a LangGraph `StateGraph`. The harness forces BM25's top 20 first
and enforces the search budget, and a forced rerank orders all candidates;
the model decides only whether and what to search. On a stratified sample of
163 development queries (all 63 hard ones, 100 others), it beat reranking by
+0.053 nDCG@10 [+0.035, +0.073], reweighted to the population. Its searches
rescued 24 of the 63 hard queries, at about 3.2× reranking's cost.

**Search agent, confirmation** ([experiments/retrieval_s3_check](experiments/retrieval_s3_check/README.md)).
The same agent, unchanged, on all 404 check queries (blind for the agent):
+0.033 nDCG@10 over reranking [+0.014, +0.053]. On queries BM25 already
handled it was slightly worse (−0.011 [−0.021, −0.002]), just past the
pre-set −0.02 harm margin. Pre-registered outcome: "confirmed with harm".
This harm did not reappear on the test split.

**Fixed reformulation** ([experiments/retrieval_s3_r1_pilot](experiments/retrieval_s3_r1_pilot/README.md)).
A harness rule replacing the agent's decision: always make exactly one
model-written search, then rerank. Paired with the agent's episodes on the
same 163 queries: −0.015 [−0.038, +0.007], inconclusive. It rescued as many
hard queries but did worse on the others, at 35% lower cost.

**Search gate** ([experiments/retrieval_s3_gate](experiments/retrieval_s3_gate/README.md)).
Free and offline, scored from recorded episodes: run the extra search only
when BM25 looks unsure. The gate selected on tune searched almost always. On
the check half (not blind) it did not remove the agent's small harm; its use
is cost, not quality.

**Configuration search** ([experiments/retrieval_config_search](experiments/retrieval_config_search/README.md)).
Free and offline: 1,089 policies that route each query, using only BM25's
scores, to the agent, the reranker or plain BM25. They cover 3 signals and
11 rates for each of the two decisions, with label-free thresholds. All were
scored from recorded outcomes on the 404 check queries. The pre-registered
rule picked the cheapest policy within 0.005 of the best: 0.816 nDCG@10 at
$0.0020 per query, against the agent's 0.819 at $0.0041. Those are selection
numbers; the policy, frozen, is P\* in the held-out evaluation above.

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
special-cases the visible test inputs. Format-2 tasks catch this with hidden
tests (two development tasks so far), and passing diffs are reviewed by hand.

**Reproducibility.** Every trace records the git commit (and whether tracked
files were modified), hashes of the prompts, tools and configuration, the
model ID the API reported, SDK and Python versions, the sandbox image ID, and
a hash of every file in the task folder.
The image's inputs are pinned, but Docker builds are not bit-for-bit
reproducible: a rebuild gets a new image ID with the same package versions.
The committed P4 and CLI runs still rescore identically on a rebuilt image.
Model outputs are stochastic (Claude Haiku 5.5 does not accept a non-default
temperature), so results are reported over repeated runs.

## What's next

The coding environment stays as a working, tested demonstration; no new coding tasks are planned for now.

SciFact's test split has now been used for its one evaluation, so new claims
need new held-out data.

1. **A second benchmark** (e.g. a multi-hop retrieval dataset), with the same
   contracts and a fresh held-out split, to see whether the findings transfer.
2. **Reducing the agent's cost:** cap the reranker's candidates, or use a
   pre-registered, cost-aware routing rule (the gate and configuration studies
   suggest most of the agent's quality at about half its cost).
3. **A larger configuration search** that includes the agent's own settings
   (search budget, results per search), run live rather than offline.
4. Later: agent-proposed harness changes, tested in isolation and reversible,
   and a transfer back to the coding environment.

oarp's configuration search is a bounded search over approved harness
settings with the model frozen. It does not demonstrate self-improvement or
general coding ability. Those are research directions, not results.
