# M2 effort pilot results

Runs: 12 (12 valid, 0 invalid). Model-configuration feasibility: effort low vs medium, 2 dev tasks x 3 repetitions each. Repetitions are clustered; no interval is given.

Estimated cost: $0.0361.

## Per task and effort

| Task | Effort | Verified success | Passed public, failed hidden | First tested candidate passed | Failing results seen | Recovered after a failure | Hit max_steps | Median model calls | Median in/out tokens | Median cost | Median wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| config_merge_007 | low | 3/3 | 0/3 | 3/3 | 0/3 | 0/3 | 0/3 | 9 | 17479/1247 | $0.0024 | 15.1 |
| config_merge_007 | medium | 3/3 | 0/3 | 3/3 | 0/3 | 0/3 | 0/3 | 10 | 19477/1405 | $0.0027 | 17.2 |
| feed_pagination_006 | low | 3/3 | 0/3 | 3/3 | 0/3 | 0/3 | 0/3 | 8 | 19253/1846 | $0.0028 | 17.0 |
| feed_pagination_006 | medium | 3/3 | 0/3 | 3/3 | 0/3 | 0/3 | 0/3 | 9 | 25236/2145 | $0.0036 | 19.5 |

## Per run

| Run | Task | Effort | Success | Public | Hidden | Stop | Calls | First tested candidate | Tool sequence |
|---|---|---|---|---|---|---|---|---|---|
| `20261022T184650-d2f5f7` | feed_pagination_006 | medium | True | passed | passed | finished | 9 | passed | read_file(README.md) > read_file(src/activity/export.py) > read_file(src/activity/cursor.py) > read_file(src/activity/store.py) > read_file(tests/test_export.py) > write_file(src/activity/cursor.py) > write_file(src/activity/store.py) > run_tests=pass |
| `20261022T184707-4931d3` | feed_pagination_006 | low | True | passed | passed | finished | 10 | passed | read_file(README.md) > read_file(src/activity/cursor.py) > read_file(src/activity/store.py) > read_file(src/activity/export.py) > read_file(tests/test_export.py) > write_file(src/activity/cursor.py) > write_file(src/activity/store.py) > write_file(src/activity/export.py) > run_tests=pass |
| `20261022T184726-6d0fe8` | config_merge_007 | medium | True | passed | passed | finished | 10 | passed | read_file(README.md) > read_file(src/appconfig/__init__.py) > read_file(src/appconfig/defaults.py) > read_file(src/appconfig/loader.py) > read_file(src/appconfig/merge.py) > read_file(src/appconfig/modes.py) > write_file(src/appconfig/loader.py) > write_file(src/appconfig/merge.py) > run_tests=pass |
| `20261022T184744-6e28b0` | config_merge_007 | low | True | passed | passed | finished | 9 | passed | read_file(README.md) > read_file(src/appconfig/defaults.py) > read_file(src/appconfig/loader.py) > read_file(src/appconfig/merge.py) > read_file(src/appconfig/modes.py) > write_file(src/appconfig/loader.py) > write_file(src/appconfig/merge.py) > run_tests=pass |
| `20261022T184802-699a5f` | feed_pagination_006 | medium | True | passed | passed | finished | 9 | passed | read_file(README.md) > read_file(src/activity/export.py) > read_file(src/activity/cursor.py) > read_file(src/activity/store.py) > read_file(tests/test_export.py) > write_file(src/activity/store.py) > write_file(src/activity/cursor.py) > run_tests=pass |
| `20261022T184825-0bc3a9` | feed_pagination_006 | low | True | passed | passed | finished | 8 | passed | read_file(README.md) > read_file(src/activity/cursor.py) > read_file(src/activity/store.py) > read_file(src/activity/export.py) > write_file(src/activity/store.py) > write_file(src/activity/cursor.py) > run_tests=pass |
| `20261022T184838-3ebf79` | config_merge_007 | medium | True | passed | passed | finished | 10 | passed | read_file(README.md) > read_file(src/appconfig/__init__.py) > read_file(src/appconfig/defaults.py) > read_file(src/appconfig/loader.py) > read_file(src/appconfig/merge.py) > read_file(src/appconfig/modes.py) > write_file(src/appconfig/merge.py) > write_file(src/appconfig/loader.py) > run_tests=pass |
| `20261022T184853-4f9daa` | config_merge_007 | low | True | passed | passed | finished | 9 | passed | read_file(README.md) > read_file(src/appconfig/defaults.py) > read_file(src/appconfig/loader.py) > read_file(src/appconfig/merge.py) > read_file(src/appconfig/modes.py) > write_file(src/appconfig/merge.py) > write_file(src/appconfig/loader.py) > run_tests=pass |
| `20261022T184908-c42094` | feed_pagination_006 | medium | True | passed | passed | finished | 10 | passed | read_file(README.md) > read_file(src/activity/export.py) > read_file(src/activity/cursor.py) > read_file(src/activity/store.py) > read_file(tests/test_export.py) > write_file(src/activity/cursor.py) > write_file(src/activity/store.py) > write_file(src/activity/export.py) > run_tests=pass |
| `20261022T184927-f10089` | feed_pagination_006 | low | True | passed | passed | finished | 8 | passed | read_file(README.md) > read_file(src/activity/export.py) > read_file(src/activity/cursor.py) > read_file(src/activity/store.py) > write_file(src/activity/store.py) > write_file(src/activity/cursor.py) > run_tests=pass |
| `20261022T184944-8bc339` | config_merge_007 | medium | True | passed | passed | finished | 10 | passed | read_file(README.md) > read_file(src/appconfig/__init__.py) > read_file(src/appconfig/defaults.py) > read_file(src/appconfig/loader.py) > read_file(src/appconfig/merge.py) > read_file(src/appconfig/modes.py) > write_file(src/appconfig/loader.py) > write_file(src/appconfig/merge.py) > run_tests=pass |
| `20261022T185001-a93bc5` | config_merge_007 | low | True | passed | passed | finished | 9 | passed | read_file(README.md) > read_file(src/appconfig/defaults.py) > read_file(src/appconfig/loader.py) > read_file(src/appconfig/merge.py) > read_file(src/appconfig/modes.py) > write_file(src/appconfig/merge.py) > write_file(src/appconfig/loader.py) > run_tests=pass |

## Failures

None.
