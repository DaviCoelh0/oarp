# S2 threshold study (paid)

**NOT BLIND: the check-half S2 curve was computed and seen before this study's contract.**

Calls: 305/305; stopped by budget: False; estimated spend $0.4124. Outcome rules: {'run': 'valid', 'H_S2': 'not shown', 'no_saving_possible': False, 'decline_warning': False, 'blindness': "NOT BLIND: the check-half S2 curve was computed and seen before this study's contract"}.

## Selection on tune (405 queries, tolerance 0.01)

tau* = 0.1801413529043867, rerank rate 0.509; nDCG@10 S0 0.668, S1 0.754, S2 0.746; S2 − S1 -0.008 [-0.015, -0.002] (n = 405).

## Confirmation on check (margin -0.02; not blind)

- S2(tau*) − S1: -0.026 [-0.039, -0.015] (n = 404)
- S2(tau*) − S0: +0.094 [+0.072, +0.116] (n = 404)
- rerank rate 0.495 (calls saved 0.505); nDCG@10 S0 0.666, S1 0.786, S2 0.760; share of S1 gain kept 0.782
- $ per query 0.00066 vs always rerank 0.00131; s per query 2.22 vs 4.19
- oracle gate: {'rerank_rate': 0.44801980198019803, 'ndcg@10': 0.7884372044673748}
- excluding 8 declined queries: -0.027 [-0.040, -0.015] (n = 396)

## Context: tau matched on tune to fixed rates, applied to check

| tune target rate | tau | check rate | check S2 nDCG@10 | S2 − S1 |
|---|---|---|---|---|
| 0.5 | 0.1770 | 0.495 | 0.760 | -0.026 [-0.039, -0.015] (n = 404) |
| 0.6 | 0.2568 | 0.629 | 0.774 | -0.012 [-0.022, -0.004] (n = 404) |
| 0.8 | 0.4571 | 0.849 | 0.786 | -0.001 [-0.002, +0.000] (n = 404) |

This study's calls: {'status_counts': {'valid': 300, 'model_error': 5}, 'declines': 5, 'decline_rate': 0.01639344262295082, 'other_error_rate': 0.0, 'estimated_usd': 0.3923686, 'mean_input_tokens': 10745.703333333333, 'mean_output_tokens': 466.65}. check-half per-query rerank costs; declined calls at the check mean $0.00131.
