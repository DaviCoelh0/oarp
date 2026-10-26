# S3 confirmation on the check half (paid)

Episodes scored: 404/404 ({'beyond': 55, 'within': 349}); smoke passed: True; stopped: False (None); estimated spend $1.7574. Outcome rules: {'run': 'valid', 'outcome': 'confirmed with harm', 'decline_warning': False}.

Blind for S3; settings frozen at the pilot's; S1 reused from the S1 check study; one episode per query.

## nDCG@10 means

| queries | S0 | S1 | S3 + forced rerank | S3 + agent ordering |
|---|---|---|---|---|
| all | 0.666 | 0.786 | 0.819 | 0.824 |
| beyond | 0.000 | 0.000 | 0.312 | 0.296 |
| within | 0.771 | 0.910 | 0.899 | 0.908 |

## Hypotheses (95% CI)

- C1 all check queries, S3+FR − S1: +0.033 [+0.014, +0.053] (n = 404)
- C2 within 20, S3+FR − S1 (margin -0.02): -0.011 [-0.021, -0.002] (n = 349)
- C3 beyond 20, S3+FR − S1: +0.312 [+0.211, +0.422] (n = 55)
- C4 S3+AO − S3+FR: overall +0.005 [-0.004, +0.016] (n = 404), beyond -0.016 [-0.053, +0.017] (n = 55), within +0.009 [-0.001, +0.020] (n = 349) → {'overall': 'inconclusive', 'beyond': 'inconclusive', 'within': 'inconclusive'}
- S3+FR − S0: +0.153 [+0.125, +0.181] (n = 404); share of the gain over S1 by stratum: {'beyond': 1.2898871901096025, 'within': -0.2898871901096026}
- tune → check: pilot reweighted to check proportions +0.049 [+0.032, +0.068] (n = {'beyond': 63, 'within': 100}), check +0.033 [+0.014, +0.053] (n = 404) (descriptive: the pilot is a stratified tune sample reweighted to check proportions)

## Mechanism (labels, analysis only)

- rescue rate (beyond: a model search added a relevant document): 0.436 (24 queries); landing in the top 10: forced rerank 1.000, agent 0.917
- harm (within: a relevant document in S1's top 10 lost): 0.011

## Model behaviour (reported separately from harness settings)

- beyond: {'episodes': 55, 'mean_searches': 1.018181818181818, 'searches': {0: 4, 1: 46, 2: 5}, 'share_no_search': 0.07272727272727272, 'mean_new_documents': 5.9818181818181815, 'mean_model_calls': 2.018181818181818}
- within: {'episodes': 349, 'mean_searches': 0.8997134670487106, 'searches': {0: 59, 1: 266, 2: 24}, 'share_no_search': 0.16905444126074498, 'mean_new_documents': 3.63323782234957, 'mean_model_calls': 1.8997134670487106}
- stop reasons {'finished': 398, 'model_error': 5, 'no_tool_call': 1}; agent ranking {'valid': 244, 'repaired': 151, 'model_error': 5, 'failed': 4}; rerank {'valid': 394, 'model_error': 9, 'repaired': 1}; rejected tool calls {}

## Cost and latency (estimates)

- $ per query: S3 0.00411 vs S1 0.00129; per stratum {'beyond': 0.004303569090909091, 'within': 0.004073899713467049}; extra $ per rescued query 0.0474; rerank share 0.373; unpriced calls 14
- s per query: S3 11.2 vs S1 4.2

## Sensitivity

- C1 excluding declined episodes: +0.034 [+0.015, +0.054] (n = 395)
- errors: {'declined_episodes': 9, 'decline_rate': 0.022277227722772276, 'other_error_episodes': 0, 'other_error_rate': 0.0}
