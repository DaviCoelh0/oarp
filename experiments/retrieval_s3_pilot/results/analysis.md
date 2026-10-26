# S3 pilot (paid)

Episodes scored: 163/163 ({'beyond': 63, 'within': 100}); smoke passed: True; stopped: False (None); estimated spend $0.7162. Outcome rules: {'run': 'valid', 'decline_warning': False, 'H1': 'positive', 'H2_holds': True, 'H3_holds': True, 'earns_confirmation': True}.

Tune half (selection data), one episode per query; S1 reused from earlier runs. Not a confirmation.

## nDCG@10 means

| stratum | S0 | S1 | S3 + forced rerank | S3 + agent ordering |
|---|---|---|---|---|
| beyond | 0.000 | 0.000 | 0.236 | 0.220 |
| within | 0.795 | 0.895 | 0.914 | 0.912 |

## Hypotheses (95% CI)

- H1 overall (stratified, tune weights {'beyond': 63, 'within': 342}), S3+FR − S1: +0.053 [+0.035, +0.073]
- H2 beyond, S3+FR − S1: +0.236 [+0.156, +0.322]
- H3 within, S3+FR − S1 (margin -0.02): +0.019 [+0.005, +0.038]
- H4 S3+AO − S3+FR: overall -0.005 [-0.020, +0.011], beyond -0.016 [-0.040, +0.007], within -0.003 [-0.021, +0.016] → {'overall': 'inconclusive', 'beyond': 'inconclusive', 'within': 'inconclusive'}

## Mechanism (labels, analysis only)

- rescue rate (beyond: a model search added a relevant document): 0.381 (24 queries); landing in the top 10: forced rerank 1.000, agent 0.917
- harm (within: a relevant document in S1's top 10 lost): forced rerank 0.000, agent 0.000
- RM3 context (beyond, same queries): first relevant in its top 10 0.032, top 20 0.159 (descriptive: RM3 is one fixed ranking; the agent adds up to 2 x 10 results)

## Model behaviour (reported separately from harness settings)

- beyond: {'episodes': 63, 'mean_searches': 1.0158730158730158, 'searches': {0: 7, 1: 48, 2: 8}, 'share_no_search': 0.1111111111111111, 'mean_new_documents': 5.603174603174603, 'mean_model_calls': 2.015873015873016}
- within: {'episodes': 100, 'mean_searches': 0.87, 'searches': {0: 23, 1: 67, 2: 10}, 'share_no_search': 0.23, 'mean_new_documents': 3.53, 'mean_model_calls': 1.87}
- stop reasons {'finished': 162, 'model_error': 1}; agent ranking {'valid': 101, 'repaired': 59, 'failed': 2, 'model_error': 1}; rerank {'valid': 161, 'model_error': 2}; rejected tool calls {}

## Cost and latency (estimates)

- $ per query (tune-weighted): S3 0.00408 vs S1 0.00129; per stratum {'beyond': 0.004322568253968254, 'within': 0.004029533}
- extra $ per rescued query 0.0470; rerank share of cost 0.375; unpriced calls 3
- s per query (tune-weighted): S3 10.3 vs S1 4.3

## Sensitivity

- H1 excluding declined episodes: +0.053 [+0.035, +0.074]
- H1 with check-half weights {'beyond': 55, 'within': 349}: +0.049 [+0.032, +0.068]
- errors: {'declined_episodes': 2, 'decline_rate': 0.012269938650306749, 'other_error_episodes': 0, 'other_error_rate': 0.0}
