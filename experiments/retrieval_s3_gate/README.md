# Gate study: search again only when BM25 looks unsure (offline, $0)

The search agent (S3) helps on queries where BM25 missed every relevant
document, but it slightly harms queries BM25 already got right
([check study](../retrieval_s3_check/README.md): "confirmed with harm").

This study asks whether a **harness gate** can fix that. The gate runs the
extra search only when BM25's own scores look unsure, and returns S1
(BM25's top 20 reranked) otherwise.

    venv/bin/python experiments/retrieval_s3_gate/analyze.py   # free: no model calls

**Why it costs nothing:** a gate only chooses, per query, between two
outcomes that are already recorded:
- S1;
- the search pipeline's outcome: the S3 agent, or R1 (one fixed reformulated
  search, see the [R1 pilot](../retrieval_s3_r1_pilot/README.md)).

R1 without an extra search is S1 by construction. So gated systems are
scored from the stored, hash-pinned records, as the S2 rule was.

Outputs: [`results/analysis.md`](results/analysis.md), `results/summary.json`
(aggregates, input hashes); per-query details in the gitignored
`results/private/`.

## Design (fixed before computing any gated number)

- **Gate:** search iff the signal < τ ("unsure"). The signal is one of three
  label-free BM25 signals: the relative gap between the top two scores (the
  S2 signal), the raw gap, or the top score.
- **Candidate gates:** 3 signals × 9 search rates (0.2 … 1.0). τ is the
  signal's quantile over all 405 tune queries at that rate; rate 1.0 means
  always search.
- **Part A (tune, selection data):**
  - Data: the 163 tune queries of the S3 and R1 pilots (63 beyond-20, 100
    within-20), reweighted to tune's proportions (63 : 342).
  - Selection rule, separately for G-S3 and G-R1: the highest stratified
    nDCG@10; ties go to the lower search rate, then to signal order.
- **Part B (check half, NOT BLIND):** the selected G-S3 gate, unchanged, on
  all 404 check queries. S3's check results were known before this
  analysis.
  - **D1:** G-S3 − S1 > 0.
  - **D2:** within-20 G-S3 − S1 > −0.02.
  - Outcome: "gate removes the harm (not blind)" if both hold, "gate does
    not help" if only D1 holds, "gate loses the gain" if D1 fails.
- **CIs:** 95% bootstraps over queries, stratified on tune.

Before the study, a free check on tune showed the signals are informative.
They separate the 63 queries BM25 missed with AUC 0.74 (relative gap), 0.78
(raw gap) and 0.79 (top score).

## Results

**Part A, tune** (selection data; the selected numbers are optimistic):

| | selected gate | search rate (tune) | nDCG@10 | always search | S1 | $ per query (gated / always / S1) |
|---|---|---|---|---|---|---|
| G-S3 | top score, rate 0.9 | 0.90 | 0.809 | 0.809 | 0.756 | 0.00394 / 0.00408 / 0.00129 |
| G-R1 | relative gap, rate 0.7 | 0.70 | 0.795 | 0.794 | 0.756 | 0.00229 / 0.00267 / 0.00129 |

- **The rule chose almost-always-search gates.** Searching helps so much on
  the beyond-20 queries that gating rarely raises nDCG@10.
- **G-S3:** keeps all 24 beyond-20 queries where search helped. Gated − S1
  is +0.053 [+0.035, +0.073]; the gate adds nothing over always searching
  (+0.000).
- **G-R1:** keeps 25 of 26. The within-20 difference against S1 shrinks
  from R1's −0.008 to −0.005 [−0.020, +0.011]. The gate saves 14% of R1's
  cost at equal nDCG@10.
- **On tune, S3 shows no within-20 harm** (+0.020 against S1), so the gate
  has nothing to remove there.

**Descriptive, not part of the selection rule:**
- G-S3 at a 50–60% search rate stays within 0.001–0.002 nDCG@10 of
  always-search while costing about 25% less ($0.00305 against $0.00408).
- G-R1 at a 50% rate does the same at $0.00197 against $0.00267.
- So on tune, the gate trades cost, not quality.

**Part B, check half** (not blind), selected G-S3 (top score, τ 22.17):

| | Difference [95% CI] |
|---|---|
| D1 G-S3 − S1 | +0.033 [+0.015, +0.053]: holds |
| D2 within-20 G-S3 − S1 (margin −0.02) | −0.010 [−0.020, −0.002]: **fails** (lower bound −0.020012) |
| G-S3 − S3 | +0.000 [−0.002, +0.003] |
| beyond-20 G-S3 − S1 | +0.312 [+0.211, +0.419] |

- **Search rate on check:** 88%.
- **Cost and latency per query:** G-S3 $0.00381 and 10.5 s; S3 $0.00411 and
  11.2 s; S1 $0.00129 and 4.2 s.
- **Pre-registered outcome: "gate does not help".** The selected gate is
  practically S3, and the within-20 loss against S1 stays at the margin.
  The lower bound misses −0.02 by 0.00001, which reads as a tie in practice,
  but the rule was fixed in advance.

## Interpretation and limitations

- **A BM25-confidence gate does not fix S3's within-20 harm.** The harm is
  not concentrated where BM25 is unsure. On tune, S3 showed none, so a gate
  selected on tune cannot learn to avoid it.
- **The gate's real value here is cost:** about a quarter of the search
  cost at almost no nDCG@10 loss on tune. That would need a pre-registered,
  cost-aware selection rule, such as fewest searches within 0.005 of
  always-search, and new evidence. It is not established by this study.
- **The remaining lever for the harm is the reranker's candidate count**
  (the deferred cap study).
- **Limits:** tune-selected gates; one recorded episode per query; records
  from different days; a non-blind check row; one dataset and one model.
