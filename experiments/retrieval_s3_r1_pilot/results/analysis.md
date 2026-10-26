# R1 pilot: reformulate once, then rerank (paid)

Episodes scored: 163/163 ({'beyond': 63, 'within': 100}); smoke passed: True; stopped: False (None); estimated spend $0.4716. Outcome rules: {'run': 'valid', 'decline_warning': False, 'primary': 'inconclusive', 'r1_within_no_harm_beyond_margin': False}.

Tune half (selection data), one episode per query, paired with the S3 pilot's episodes and the reused S1 records for the same queries.

## nDCG@10 means

| stratum | S0 | S1 | S3 (pilot) | R1 |
|---|---|---|---|---|
| beyond | 0.000 | 0.000 | 0.236 | 0.289 |
| within | 0.795 | 0.895 | 0.914 | 0.887 |

## Comparisons (95% CI)

- **primary** R1 − S3 (stratified, tune weights {'beyond': 63, 'within': 342}): -0.015 [-0.038, +0.007] → {'reading': 'inconclusive', 's3_decision_adds_value': False, 'r1_better': False, 'equivalent_within_band': False}
- R1 − S3 by stratum: beyond +0.054 [-0.018, +0.133], within -0.027 [-0.051, -0.005]
- R1 − S1: overall +0.038 [+0.018, +0.059], beyond +0.289 [+0.200, +0.384], within -0.008 [-0.025, +0.009] (harm margin −0.02)

## Mechanism (labels, analysis only)

- rescue rate (beyond): R1 0.413, S3 0.381
- harm (within: a relevant document in S1's top 10 lost): R1 0.010, S3 0.000
- new documents per episode: {'beyond': {'r1': 6.349206349206349, 's3': 5.603174603174603}, 'within': {'r1': 4.38, 's3': 3.53}}; S3 searched at all: {'beyond': 0.8888888888888888, 'within': 0.77}
- overlap of added documents (Jaccard, R1 vs S3's first search): 0.211 over 156 episodes
- behaviour: {'reformulation_status': {'model_error': 2, 'valid': 161}, 'rerank_status': {'model_error': 2, 'valid': 161}, 'rejected_tool_calls': {}}

## Cost and latency (estimates, tune-weighted per query)

- $: R1 0.00267, S3 0.00408, S1 0.00129; unpriced calls 4
- s: R1 7.5, S3 10.3, S1 4.3

## Sensitivity

- primary excluding declined episodes: -0.015 [-0.038, +0.007]
- errors: {'declined_episodes': 2, 'decline_rate': 0.012269938650306749, 'other_error_episodes': 0, 'other_error_rate': 0.0}
