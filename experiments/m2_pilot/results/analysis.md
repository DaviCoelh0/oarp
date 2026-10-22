# M2 effort pilot results

Runs: 6 (6 valid, 0 invalid). Model-configuration feasibility: effort low vs medium, 2 dev tasks x 3 repetitions each. Repetitions are clustered; no interval is given.

Estimated cost: $0.0208.

## Per task and effort

| Task | Effort | Verified success | Passed public, failed hidden | First tested candidate passed | Failing results seen | Recovered after a failure | Hit max_steps | Median model calls | Median in/out tokens | Median cost | Median wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| config_merge_007 | medium | 3/3 | 0/3 | 3/3 | 0/3 | 0/3 | 0/3 | 10 | 19593/1457 | $0.0027 | 15.5 |
| feed_pagination_006 | medium | 3/3 | 0/3 | 3/3 | 0/3 | 0/3 | 0/3 | 9 | 25554/2376 | $0.0037 | 20.9 |

## Per run

| Run | Task | Effort | Success | Public | Hidden | Stop | Calls | First tested candidate | Tool sequence |
|---|---|---|---|---|---|---|---|---|---|
| `20261022T181702-0c2d84` | feed_pagination_006 | medium | True | passed | passed | finished | 9 | passed | read_file(README.md) > read_file(src/activity/export.py) > read_file(src/activity/cursor.py) > read_file(src/activity/store.py) > read_file(tests/test_export.py) > write_file(src/activity/store.py) > write_file(src/activity/cursor.py) > run_tests=pass |
| `20261022T181722-78573b` | config_merge_007 | medium | True | passed | passed | finished | 10 | passed | read_file(README.md) > read_file(src/appconfig/__init__.py) > read_file(src/appconfig/defaults.py) > read_file(src/appconfig/loader.py) > read_file(src/appconfig/merge.py) > read_file(src/appconfig/modes.py) > write_file(src/appconfig/merge.py) > write_file(src/appconfig/loader.py) > run_tests=pass |
| `20261022T181738-394681` | feed_pagination_006 | medium | True | passed | passed | finished | 9 | passed | read_file(README.md) > read_file(src/activity/cursor.py) > read_file(src/activity/export.py) > read_file(src/activity/store.py) > read_file(tests/test_export.py) > write_file(src/activity/store.py) > write_file(src/activity/cursor.py) > run_tests=pass |
| `20261022T181759-df24c7` | config_merge_007 | medium | True | passed | passed | finished | 10 | passed | read_file(README.md) > read_file(src/appconfig/__init__.py) > read_file(src/appconfig/defaults.py) > read_file(src/appconfig/loader.py) > read_file(src/appconfig/merge.py) > read_file(src/appconfig/modes.py) > write_file(src/appconfig/merge.py) > write_file(src/appconfig/loader.py) > run_tests=pass |
| `20261022T181815-a448e8` | feed_pagination_006 | medium | True | passed | passed | finished | 11 | passed | read_file(README.md) > read_file(src/activity/export.py) > read_file(src/activity/cursor.py) > read_file(src/activity/store.py) > read_file(tests/test_export.py) > write_file(src/activity/store.py) > write_file(src/activity/cursor.py) > read_file(src/activity/__init__.py) > write_file(tests/test_export.py)!refused > run_tests=pass |
| `20261022T181838-246da7` | config_merge_007 | medium | True | passed | passed | finished | 10 | passed | read_file(README.md) > read_file(src/appconfig/__init__.py) > read_file(src/appconfig/defaults.py) > read_file(src/appconfig/loader.py) > read_file(src/appconfig/merge.py) > read_file(src/appconfig/modes.py) > write_file(src/appconfig/merge.py) > write_file(src/appconfig/loader.py) > run_tests=pass |

## Failures

None.
