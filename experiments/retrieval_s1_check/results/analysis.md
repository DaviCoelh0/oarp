# S1 check-half confirmation (paid)

Calls: 444/444; stopped by budget: False; estimated spend $0.6084. Queries: 404. Outcome rules: {'run': 'valid', 'H1c': 'confirmed', 'decline_warning': False, 'cost_surprise': False}.

- H1c S1-plain − S0: +0.120 [+0.096, +0.144] (n = 404)
- Excluding 8 declined queries (pre-specified): +0.122 [+0.098, +0.148] (n = 396)

| | S0 | S1-plain | perfect top-20 |
|---|---|---|---|
| nDCG@10 | 0.666 | 0.786 | 0.851 |
| Recall@10 | 0.789 | 0.841 | |
| MRR@10 | 0.633 | 0.778 | |

Share of ceiling 0.651; improved / harmed / unchanged [113, 14, 277]; harm rate 0.035 (rank 1 relevant: 0.027).
By bucket: {'rank 1': {'queries': 223, 's0': 0.9895813559736364, 's1': 0.9853515808175124}, 'ranks 2-10': {'queries': 103, 's0': 0.4700049405934831, 's1': 0.7852939252955508}, 'ranks 11-20': {'queries': 23, 's0': 0.0, 's1': 0.7376832137937724}, 'beyond 20': {'queries': 55, 's0': 0.0, 's1': 0.0}}.
Status {'valid': 435, 'model_error': 9}; declines 9 (0.020, {'bio': 9}); other model errors 0 (0.000); repairs {'duplicates': 0, 'out_of_range': 0, 'missing': 0}; truncations 0.
Tokens in {'mean': 10844.636781609195, 'p90': 12625}, out {'mean': 462.69885057471265, 'p90': 613}; est. $ {'total': 0.5723787, 'per_call_mean': 0.0013158131034482759}; latency {'median': 4.273652604431845, 'p90': 5.716912958072498}.
Stability: {'queries': 20, 'mean_sd_ndcg@10': 0.003512835600102934, 'queries_with_varying_ndcg': 2, 'top1_agreement': 0.75, 'mean_kendall_tau_top20': 0.7607017543859649} (pilot: {'queries': 20, 'mean_sd_ndcg@10': 0.0, 'top1_agreement': 0.95, 'mean_kendall_tau_top20': 0.7803508771929825}).
For gate studies: {'oracle_gate_rate': 0.44801980198019803, 'oracle_gate_x_s1_ndcg@10': 0.7884372044673748, 's2_x_s1_curve': [{'target_rate': 0.2, 'tau': 0.05444823410377594, 'rate': 0.2004950495049505, 'ndcg@10': 0.7106395487270715}, {'target_rate': 0.3, 'tau': 0.09294272831752023, 'rate': 0.2995049504950495, 'ndcg@10': 0.7369522298359874}, {'target_rate': 0.4, 'tau': 0.13063908566689825, 'rate': 0.39851485148514854, 'ndcg@10': 0.7452501088510515}, {'target_rate': 0.5, 'tau': 0.183997785523257, 'rate': 0.5, 'ndcg@10': 0.7599862369857308}, {'target_rate': 0.6, 'tau': 0.23798570807594738, 'rate': 0.599009900990099, 'ndcg@10': 0.7725050769080936}, {'target_rate': 0.8, 'tau': 0.421778574341763, 'rate': 0.7995049504950495, 'ndcg@10': 0.7850511215282121}]}.
