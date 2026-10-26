# Configuration search over BM25-routed retrieval policies (offline, $0)

1089 policies (591 distinct outcomes) scored on 404 check queries; thresholds from 809 development queries. Selection data: the selected policy's numbers here are optimistic.

| policy | nDCG@10 | $ per query | s per query | share S3 | share S1 |
|---|---|---|---|---|---|
| S0 | 0.666 | 0.00000 | 0.0 | 0.00 | 0.00 |
| S1 | 0.786 | 0.00131 | 4.2 | 0.00 | 1.00 |
| S3 | 0.819 | 0.00411 | 11.2 | 1.00 | 0.00 |
| best | 0.821 | 0.00325 | 9.1 | 0.70 | 0.24 |
| **selected** | 0.816 | 0.00196 | 5.8 | 0.29 | 0.53 |

Selected (cheapest within 0.005 of best): search if top_score < 9.8509 (rate 0.3), else rerank if raw_gap < 7.0108 (rate 0.8), else BM25.
- selected − S1 +0.030 [+0.013, +0.048]; − S3 -0.003 [-0.013, +0.006]; − S0 +0.150 [+0.123, +0.178] (check, optimistic)
- tune cross-check (stratified, descriptive): selected 0.790 at $0.00211; S1 0.756; S3 0.809

## Pareto frontier on check (cost ascending)

| $ per query | nDCG@10 | share S3 | share S1 | policy |
|---|---|---|---|---|
| 0.00000 | 0.666 | 0.00 | 0.00 | search if relative_gap < -inf (rate 0.0), else rerank if relative_gap < -inf (rate 0.0), else BM25 |
| 0.00011 | 0.691 | 0.00 | 0.09 | search if relative_gap < -inf (rate 0.0), else rerank if top_score < 6.7309 (rate 0.1), else BM25 |
| 0.00014 | 0.693 | 0.00 | 0.10 | search if relative_gap < -inf (rate 0.0), else rerank if raw_gap < 0.2388 (rate 0.1), else BM25 |
| 0.00024 | 0.710 | 0.00 | 0.19 | search if relative_gap < -inf (rate 0.0), else rerank if top_score < 8.1424 (rate 0.2), else BM25 |
| 0.00026 | 0.711 | 0.00 | 0.20 | search if relative_gap < -inf (rate 0.0), else rerank if relative_gap < 0.0524 (rate 0.2), else BM25 |
| 0.00026 | 0.713 | 0.00 | 0.20 | search if relative_gap < -inf (rate 0.0), else rerank if raw_gap < 0.4796 (rate 0.2), else BM25 |
| 0.00037 | 0.731 | 0.00 | 0.28 | search if relative_gap < -inf (rate 0.0), else rerank if relative_gap < 0.0826 (rate 0.3), else BM25 |
| 0.00048 | 0.743 | 0.00 | 0.37 | search if relative_gap < -inf (rate 0.0), else rerank if raw_gap < 1.3623 (rate 0.4), else BM25 |
| 0.00051 | 0.745 | 0.00 | 0.38 | search if relative_gap < -inf (rate 0.0), else rerank if relative_gap < 0.1231 (rate 0.4), else BM25 |
| 0.00064 | 0.763 | 0.00 | 0.49 | search if relative_gap < -inf (rate 0.0), else rerank if raw_gap < 2.1810 (rate 0.5), else BM25 |
| 0.00079 | 0.772 | 0.00 | 0.60 | search if relative_gap < -inf (rate 0.0), else rerank if raw_gap < 3.2119 (rate 0.6), else BM25 |
| 0.00081 | 0.773 | 0.00 | 0.61 | search if relative_gap < -inf (rate 0.0), else rerank if relative_gap < 0.2448 (rate 0.6), else BM25 |
| 0.00094 | 0.783 | 0.00 | 0.72 | search if relative_gap < -inf (rate 0.0), else rerank if raw_gap < 4.6628 (rate 0.7), else BM25 |
| 0.00096 | 0.784 | 0.00 | 0.72 | search if relative_gap < -inf (rate 0.0), else rerank if relative_gap < 0.3448 (rate 0.7), else BM25 |
| 0.00108 | 0.786 | 0.00 | 0.82 | search if relative_gap < -inf (rate 0.0), else rerank if relative_gap < 0.4345 (rate 0.8), else BM25 |
| 0.00108 | 0.786 | 0.00 | 0.82 | search if relative_gap < -inf (rate 0.0), else rerank if raw_gap < 7.0108 (rate 0.8), else BM25 |
| 0.00120 | 0.790 | 0.09 | 0.63 | search if top_score < 6.7309 (rate 0.1), else rerank if raw_gap < 4.6628 (rate 0.7), else BM25 |
| 0.00122 | 0.790 | 0.09 | 0.64 | search if top_score < 6.7309 (rate 0.1), else rerank if relative_gap < 0.3448 (rate 0.7), else BM25 |
| 0.00134 | 0.792 | 0.09 | 0.73 | search if top_score < 6.7309 (rate 0.1), else rerank if relative_gap < 0.4345 (rate 0.8), else BM25 |
| 0.00135 | 0.793 | 0.09 | 0.73 | search if top_score < 6.7309 (rate 0.1), else rerank if raw_gap < 7.0108 (rate 0.8), else BM25 |
| 0.00148 | 0.793 | 0.09 | 0.84 | search if top_score < 6.7309 (rate 0.1), else rerank if relative_gap < 0.5302 (rate 0.9), else BM25 |
| 0.00150 | 0.796 | 0.20 | 0.52 | search if relative_gap < 0.0524 (rate 0.2), else rerank if raw_gap < 4.6628 (rate 0.7), else BM25 |
| 0.00151 | 0.797 | 0.20 | 0.52 | search if relative_gap < 0.0524 (rate 0.2), else rerank if relative_gap < 0.3448 (rate 0.7), else BM25 |
| 0.00152 | 0.799 | 0.19 | 0.53 | search if top_score < 8.1424 (rate 0.2), else rerank if raw_gap < 4.6628 (rate 0.7), else BM25 |
| 0.00155 | 0.800 | 0.19 | 0.54 | search if top_score < 8.1424 (rate 0.2), else rerank if relative_gap < 0.3448 (rate 0.7), else BM25 |
| 0.00166 | 0.802 | 0.19 | 0.64 | search if top_score < 8.1424 (rate 0.2), else rerank if relative_gap < 0.4345 (rate 0.8), else BM25 |
| 0.00166 | 0.802 | 0.19 | 0.64 | search if top_score < 8.1424 (rate 0.2), else rerank if raw_gap < 7.0108 (rate 0.8), else BM25 |
| 0.00173 | 0.805 | 0.29 | 0.36 | search if top_score < 9.8509 (rate 0.3), else rerank if relative_gap < 0.2448 (rate 0.6), else BM25 |
| 0.00182 | 0.813 | 0.29 | 0.43 | search if top_score < 9.8509 (rate 0.3), else rerank if raw_gap < 4.6628 (rate 0.7), else BM25 |
| 0.00186 | 0.814 | 0.29 | 0.47 | search if top_score < 9.8509 (rate 0.3), else rerank if relative_gap < 0.3448 (rate 0.7), else BM25 |
| 0.00196 | 0.816 | 0.29 | 0.53 | search if top_score < 9.8509 (rate 0.3), else rerank if raw_gap < 7.0108 (rate 0.8), else BM25 |
| 0.00209 | 0.816 | 0.29 | 0.64 | search if top_score < 9.8509 (rate 0.3), else rerank if relative_gap < 0.5302 (rate 0.9), else BM25 |
| 0.00217 | 0.816 | 0.39 | 0.37 | search if top_score < 11.2541 (rate 0.4), else rerank if relative_gap < 0.3448 (rate 0.7), else BM25 |
| 0.00225 | 0.817 | 0.39 | 0.43 | search if top_score < 11.2541 (rate 0.4), else rerank if raw_gap < 7.0108 (rate 0.8), else BM25 |
| 0.00239 | 0.818 | 0.39 | 0.54 | search if top_score < 11.2541 (rate 0.4), else rerank if relative_gap < 0.5302 (rate 0.9), else BM25 |
| 0.00308 | 0.819 | 0.70 | 0.11 | search if top_score < 16.2298 (rate 0.7), else rerank if raw_gap < 4.6628 (rate 0.7), else BM25 |
| 0.00308 | 0.820 | 0.70 | 0.11 | search if top_score < 16.2298 (rate 0.7), else rerank if relative_gap < 0.2448 (rate 0.6), else BM25 |
| 0.00313 | 0.820 | 0.70 | 0.15 | search if top_score < 16.2298 (rate 0.7), else rerank if relative_gap < 0.3448 (rate 0.7), else BM25 |
| 0.00325 | 0.821 | 0.70 | 0.24 | search if top_score < 16.2298 (rate 0.7), else rerank if relative_gap < 0.5302 (rate 0.9), else BM25 |

Unpriced: {'s1_check': 8, 's3_check_calls': 14}; S1 declines priced at $0.00131.
