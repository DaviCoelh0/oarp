# S1 pilot (paid)

Calls: 280/280; stopped by budget: False; estimated spend $0.3830. Queries: 100. Outcome rules: {'run': 'valid', 'format_problem': False, 'H1': 'positive', 'H2': 'inconclusive', 'cost_surprise': False}.

## Paired nDCG@10 differences (pass 1, bootstrap 95% CI)

- H1 S1-plain − S0: +0.056 [+0.021, +0.095] (n = 100)
- H2 S1-shuffled − S1-plain: -0.004 [-0.013, +0.002] (n = 100)
- S1-shuffled − S0: +0.052 [+0.014, +0.093] (n = 100)

## Per condition

| | plain | shuffled |
|---|---|---|
| nDCG@10 | 0.754 | 0.750 |
| Recall@10 | 0.820 | 0.820 |
| MRR@10 | 0.730 | 0.724 |
| S0 nDCG@10 | 0.698 | 0.698 |
| perfect top-20 nDCG@10 | 0.830 | 0.830 |
| share of ceiling | 0.425 | 0.393 |
| harm rate | 0.050 | 0.060 |
| harm rate (rank 1 relevant) | 0.032 | 0.048 |
| copied presentation order | 0.000 | 0.000 |
| final top 20 = BM25 | 0.030 | 0.040 |
| Spearman shown vs model rank | 0.435 | 0.057 |
| model-error rate | 0.021 | 0.029 |
| format-problem rate | 0.000 | 0.000 |
| truncations | 0 | 0 |
| status counts | {'valid': 137, 'model_error': 3} | {'valid': 136, 'model_error': 4} |
| input tokens mean / p90 | 10860 / 12628 | 10866 / 12628 |
| output tokens mean / p90 | 462 / 584 | 446 / 578 |
| est. $ total / per call | 0.1805 / 0.00132 | 0.1781 / 0.00131 |
| latency s median / p90 | 4.4 / 5.6 | 4.2 / 5.6 |
| stability | {'queries': 20, 'mean_sd_ndcg@10': 0.0, 'top1_agreement': 0.95, 'mean_kendall_tau_top20': 0.7803508771929825} | {'queries': 20, 'mean_sd_ndcg@10': 0.0, 'top1_agreement': 0.95, 'mean_kendall_tau_top20': 0.7631578947368421} |
| exploratory gates | {'oracle_gate_rate': 0.38, 'oracle_gate_x_s1_ndcg@10': 0.7612517267809707, 's2_tau': 0.10422520508698095, 's2_rate': 0.38, 's2_x_s1_ndcg@10': 0.7468939151490752} | {'oracle_gate_rate': 0.38, 'oracle_gate_x_s1_ndcg@10': 0.7620304327229729, 's2_tau': 0.10422520508698095, 's2_rate': 0.38, 's2_x_s1_ndcg@10': 0.7475099782655238} |

Bucket mix (sample vs tune half): {'sample': {'rank 1': 0.62, 'ranks 2-10': 0.17, 'ranks 11-20': 0.04, 'beyond 20': 0.17}, 'tune_half': {'rank 1': 0.5654320987654321, 'ranks 2-10': 0.23209876543209876, 'ranks 11-20': 0.04691358024691358, 'beyond 20': 0.15555555555555556}}
R2a cost estimate for comparison: {'mean_prompt_chars': 30249, 'est_input_tokens': 7562, 'per_query_usd': {'low': [0.0007862, 0.0010362000000000001], 'medium': [0.0008862000000000001, 0.0017862]}}
