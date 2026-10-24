# S2b gate pilot (paid)

Calls: 600/600; stopped by budget: False; estimated spend $0.1552. Queries: 100; oracle rerank rate 0.380. Outcome rules: {'run': 'valid', 'format_problem': False, 'primary': 'gate not better', 'no_saving': False, 'unstable': False}.

- Primary, S2b-medium − S2 (matched rate), share of right decisions: +0.060 [-0.050, +0.180] (n = 100)
- S2b-low − S2 (matched): +0.060 [-0.050, +0.180] (n = 100)
- low − medium, right decisions: -0.010 [-0.050, +0.020] (n = 100)

| gate | rerank rate | right decisions | S2 right (matched) | precision | recall | flip rate | unparseable (pass 1) | declines | $ per call | median s |
|---|---|---|---|---|---|---|---|---|---|---|
| medium | 0.340 | 0.800 | 0.740 | 0.818 | 0.871 | 0.020 | 0.000 | 6 | 0.00024 | 2.2 |
| low | 0.370 | 0.790 | 0.730 | 0.825 | 0.839 | 0.040 | 0.000 | 3 | 0.00021 | 1.0 |

| strategy | rerank rate | nDCG@10 | − S1 (95% CI) | share of S1 gain | $ per query | s per query |
|---|---|---|---|---|---|---|
| S0 | 0.000 | 0.698 | -0.056 [-0.095, -0.021] (n = 100) | 0.000 | 0.00000 | 0.0 |
| S1 | 1.000 | 0.754 | +0.000 [+0.000, +0.000] (n = 100) | 1.000 | 0.00130 | 4.4 |
| oracle_gate | 0.380 | 0.761 | +0.007 [+0.000, +0.018] (n = 100) | 1.131 | 0.00049 | 1.7 |
| S2b_medium | 0.340 | 0.729 | -0.025 [-0.049, -0.007] (n = 100) | 0.551 | 0.00067 | 3.6 |
| S2_matched_medium | 0.340 | 0.736 | -0.018 [-0.042, +0.003] (n = 100) | 0.689 | 0.00043 | 1.5 |
| S2b_low | 0.370 | 0.729 | -0.025 [-0.049, -0.007] (n = 100) | 0.551 | 0.00067 | 3.0 |
| S2_matched_low | 0.370 | 0.740 | -0.014 [-0.037, +0.005] (n = 100) | 0.754 | 0.00047 | 1.6 |

S1 pilot rerank cost per query; declined pilot calls use the pilot mean $0.00130
