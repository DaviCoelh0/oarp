# Gate study: search only when BM25 looks unsure (offline, $0)

Inputs pinned: True. Tune sample 163 queries (weights {'beyond': 63, 'within': 342}), thresholds from 405 tune queries; check 404 queries. One recorded episode per query.

## Part A, tune (selection data): G-S3 (gate the agent)

| signal | rate 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| relative_gap | 0.769 | 0.779 | 0.795 | 0.797 | 0.805 | 0.809 | 0.809 | 0.809 | 0.809 |
| raw_gap | 0.771 | 0.786 | 0.793 | 0.799 | 0.808 | 0.809 | 0.809 | 0.809 | 0.809 |
| top_score | 0.780 | 0.787 | 0.797 | 0.797 | 0.801 | 0.807 | 0.809 | 0.809 | 0.809 |

Selected: top_score at target rate 0.9 (tau 22.1736, achieved 0.899 of tune, 0.941 weighted in the sample). nDCG@10 0.809 (always search 0.809, S1 0.756); within 0.915, beyond 0.236.
- gated − S1: +0.053 [+0.035, +0.073]; gated − always search: +0.000 [+0.000, +0.001]
- within gated − S1: +0.020 [+0.005, +0.038]; beyond gated − S1: +0.236 [+0.155, +0.320]
- beyond queries where search helped: 24, kept by the gate: 24
- $ per query: gated 0.00394, always search 0.00408, S1 0.00129; s per query gated 10.1, always 10.3
- optimistic: chosen on the same queries

## Part A, tune (selection data): G-R1 (gate the fixed reformulation)

| signal | rate 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| relative_gap | 0.772 | 0.780 | 0.790 | 0.794 | 0.793 | 0.795 | 0.794 | 0.794 | 0.794 |
| raw_gap | 0.776 | 0.784 | 0.787 | 0.791 | 0.792 | 0.793 | 0.793 | 0.794 | 0.794 |
| top_score | 0.774 | 0.782 | 0.786 | 0.786 | 0.788 | 0.787 | 0.790 | 0.795 | 0.794 |

Selected: relative_gap at target rate 0.7 (tau 0.3557, achieved 0.699 of tune, 0.744 weighted in the sample). nDCG@10 0.795 (always search 0.794, S1 0.756); within 0.890, beyond 0.281.
- gated − S1: +0.040 [+0.020, +0.059]; gated − always search: +0.001 [-0.004, +0.008]
- within gated − S1: -0.005 [-0.020, +0.011]; beyond gated − S1: +0.281 [+0.190, +0.376]
- beyond queries where search helped: 26, kept by the gate: 25
- $ per query: gated 0.00229, always search 0.00267, S1 0.00129; s per query gated 6.8, always 7.5
- optimistic: chosen on the same queries

## Part B, check half: the selected G-S3 gate unchanged (NOT BLIND: S3's check results (overall and per stratum) were known before this analysis)

Signal top_score, tau 22.1736; search rate on check 0.884. nDCG@10: gated 0.819, S1 0.786, S3 0.819.
- D1 gated − S1: +0.033 [+0.015, +0.053]
- D2 within gated − S1 (margin -0.02): -0.010 [-0.020, -0.002]
- gated − S3: +0.000 [-0.002, +0.003]; beyond gated − S1: +0.312 [+0.211, +0.419]
- $ per query: gated 0.00381, S3 0.00411, S1 0.00129; s per query gated 10.5, S3 11.2, S1 4.2
- outcome: **gate does not help**
