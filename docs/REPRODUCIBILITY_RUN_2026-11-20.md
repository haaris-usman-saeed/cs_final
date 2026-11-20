# Clean-checkout reproduction run — 2026-11-20

* **Machine:** macOS arm64, Python 3.14.5.
* **Source:** a fresh `git clone --branch dev/v0.3-multiregion` of https://github.com/haaris-usman-saeed/cs_final at
  commit `3b97017`, a new virtualenv, and **no cached data**: all 409 raw files were downloaded again, with no API keys.
* **Script:** the exact script used is below. Its unedited log follows.

The run passed on every check:
- all 34 tests passed;
- the rebuilt dataset content is identical to the released file;
- the same champion was selected;
- the validation leaderboard is identical (max |ΔAP| = 0.0 over 21 rows);
- the one-time-test APs are identical in all four test sets.

Total wall time was 1,730 s (~29 min): ~15 min ingest, ~9 min development stage, ~3 min test.

Running `evaluate-test-v3` here reproduces the already-published test numbers in a separate output directory. It is a
reproduction, **not** a new or additional evaluation.

## Script
```bash
#!/usr/bin/env bash
# Clean-checkout reproduction of FireRisk Open v0.3 (fresh clone, fresh venv, no cached data).
set -uo pipefail
R="$1"; cd "$R"
t0=$(date +%s); step(){ echo "=== [$(( $(date +%s)-t0 ))s] $*"; }
step clone; rm -rf cs_final && git clone -q --branch dev/v0.3-multiregion https://github.com/haaris-usman-saeed/cs_final.git && cd cs_final && git log --oneline -1
step venv; python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev]" 2>&1 | grep -v -i notice | tail -2
step xgboost-fix; scripts/macos_xgboost_libomp.sh 2>&1 | tail -1
step pytest; .venv/bin/python -m pytest -q -W ignore 2>&1 | tail -1
step ingest; .venv/bin/python -W ignore -m fireaq ingest-v3 > ingest.log 2>&1; echo "exit $?"; tail -3 ingest.log
step build; .venv/bin/python -W ignore -m fireaq build-v3 > build.log 2>&1; echo "exit $?"
step compare-dataset; .venv/bin/python - <<'PY'
import pandas as pd
a=pd.read_parquet('data/v03/dataset/firerisk_open_multiregion_v0.3.0.parquet'); b=pd.read_parquet('release_v3/firerisk_open_multiregion_v0.3.0.parquet')
cols=[c for c in a.columns if c!='raw_manifest_sha256']
print('shape', a.shape, b.shape, 'content identical (excluding manifest hash col):', a[cols].equals(b[cols]))
print('raw_manifest_sha256 differs (retrieval times differ):', a.raw_manifest_sha256.iloc[0]!=b.raw_manifest_sha256.iloc[0])
PY
step experiment; .venv/bin/python -W ignore -m fireaq experiment-v3 --out output_repro > experiment.log 2>&1; echo "exit $?"; grep CHAMPION experiment.log
step compare-leaderboard; .venv/bin/python - <<'PY'
import pandas as pd, numpy as np
a=pd.read_csv('output_repro/validation_leaderboard_v3.csv'); b=pd.read_csv('release_v3/validation_leaderboard_v3.csv')
m=a.merge(b,on=['model','feature_set'],suffixes=('_repro','_rel'))
d=(m.oof_ap_pooled_repro-m.oof_ap_pooled_rel).abs()
print('rows',len(m),'max |dAP| pooled OOF',float(d.max()),'max |dLORO|',float((m.loro_ap_pooled_repro-m.loro_ap_pooled_rel).abs().max()))
PY
step evaluate; .venv/bin/python -W ignore -m fireaq evaluate-test-v3 --out output_repro > evaluate.log 2>&1; echo "exit $?"; .venv/bin/python - <<'PY'
import json
a=json.load(open('output_repro/test_results_v3.json')); b=json.load(open('release_v3/test_results_v3.json'))
for r in ['test_time','test_space_primary','test_space_secondary','previously_tested_norcal']:
    print(r, 'AP repro', a[r]['metrics']['champion']['average_precision'], 'released', b[r]['metrics']['champion']['average_precision'])
PY
step done
```

## Log
```
=== [0s] clone
3b97017 v0.3: measure POWER MERRA-2 -> GEOS-IT live-weather shift and its effect on the frozen model
=== [5s] venv

=== [20s] xgboost-fix
xgboost 3.4.1 loads OK
=== [44s] pytest
34 passed in 40.90s
=== [86s] ingest
exit 0
  "raw_files": 409,
  "network_downloads_this_run": 409
}
=== [979s] build
exit 0
=== [998s] compare-dataset
shape (973836, 65) (973836, 65) content identical (excluding manifest hash col): True
raw_manifest_sha256 differs (retrieval times differ): True
=== [999s] experiment
exit 0
CHAMPION extra_trees|B_plus_fire_history OOF AP=0.1595 LORO AP=0.1684 (1SE=0.0157)
=== [1556s] compare-leaderboard
rows 21 max |dAP| pooled OOF 0.0 max |dLORO| 0.0
=== [1556s] evaluate
exit 0
test_time AP repro 0.18563119502654407 released 0.18563119502654407
test_space_primary AP repro 0.1298410255942653 released 0.1298410255942653
test_space_secondary AP repro 0.12045501888041238 released 0.12045501888041238
previously_tested_norcal AP repro 0.09356376939495989 released 0.09356376939495989
=== [1730s] done
```
