# P4 feasibility results

Runs: 15 (15 valid, 0 invalid). Preliminary feasibility study: 5 dev tasks, small n; not a benchmark.

**Overall verified success: 15/15** (Wilson 95% interval 0.80-1.00). Estimated cost: $0.0173.

| Task | Success | 95% CI | Stop reasons | Median model calls | Median tool calls | Used tests | Errors | Recovered | Mean in/out tokens | Cost | Median wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cart_rounding_004 | 3/3 | 0.44-1.00 | finished x3 | 6 | 5 | 3/3 | none | 0 | 10374/886 | $0.0044 | 9.007 |
| lru_cache_005 | 3/3 | 0.44-1.00 | finished x3 | 5 | 4 | 3/3 | none | 0 | 8796/787 | $0.0038 | 7.561 |
| merge_intervals_003 | 3/3 | 0.44-1.00 | finished x3 | 5 | 4 | 3/3 | none | 0 | 7380/705 | $0.0033 | 6.615 |
| mutable_default_002 | 3/3 | 0.44-1.00 | finished x3 | 5 | 4 | 3/3 | none | 0 | 6446/621 | $0.0029 | 6.647 |
| off_by_one_001 | 3/3 | 0.44-1.00 | finished x3 | 5 | 4 | 3/3 | none | 0 | 6506/650 | $0.0029 | 6.554 |

Model calls with a reasoning summary: 19/78.

## Failures

None.

## Passing runs: diffs to review by hand (gaming check)

- `20261021T222415-1d31b1` cart_rounding_004: `20261021T222415-1d31b1/final.diff`
- `20261021T222424-ec7ff5` lru_cache_005: `20261021T222424-ec7ff5/final.diff`
- `20261021T222431-e611eb` merge_intervals_003: `20261021T222431-e611eb/final.diff`
- `20261021T222438-c151e9` mutable_default_002: `20261021T222438-c151e9/final.diff`
- `20261021T222444-f49697` off_by_one_001: `20261021T222444-f49697/final.diff`
- `20261021T222451-8972e9` cart_rounding_004: `20261021T222451-8972e9/final.diff`
- `20261021T222500-c60590` lru_cache_005: `20261021T222500-c60590/final.diff`
- `20261021T222507-a0a52f` merge_intervals_003: `20261021T222507-a0a52f/final.diff`
- `20261021T222514-54af07` mutable_default_002: `20261021T222514-54af07/final.diff`
- `20261021T222520-bc2755` off_by_one_001: `20261021T222520-bc2755/final.diff`
- `20261021T222527-5a2444` cart_rounding_004: `20261021T222527-5a2444/final.diff`
- `20261021T222536-edf156` lru_cache_005: `20261021T222536-edf156/final.diff`
- `20261021T222543-a548c6` merge_intervals_003: `20261021T222543-a548c6/final.diff`
- `20261021T222550-57238a` mutable_default_002: `20261021T222550-57238a/final.diff`
- `20261021T222557-6655ff` off_by_one_001: `20261021T222557-6655ff/final.diff`
