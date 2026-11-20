#!/usr/bin/env bash
# FireRisk Open daily job: ingest NRT + weather, score with the frozen v0.3 model, log prospectively, publish safely.
# Daily INFERENCE only — it never retrains or re-selects a model.
#
# Env:  STATE_DIR (default: state)    persistent state (NRT store, prospective log, public files)
#       DRY_RUN=1                      run everything but do not publish predictions
#       REGIONS=a,b                    subset of regions (default: all six)
#       FIRMS_MAP_KEY                  optional; enables NRT backfill through the FIRMS API
# Exit: 0 published OK | 3 run refused by a safeguard (nothing new published) | 2 model verification failed | other = crash
set -uo pipefail
cd "$(dirname "$0")/.."
STATE_DIR=${STATE_DIR:-state}; PY=${PYTHON:-python}
mkdir -p "$STATE_DIR/live" "$STATE_DIR/prospective" "$STATE_DIR/public/data" "$STATE_DIR/staging" "$STATE_DIR/manifests"
today=$(date -u +%F)
log(){ echo "[daily $(date -u +%H:%M:%SZ)] $*"; }

# 1. model: rebuild from the committed dataset and verify the frozen-model fingerprint (refuse on mismatch)
if [ ! -f output_v3/model_v3.joblib ]; then
  log "rebuilding frozen champion"; $PY -W ignore -m fireaq rebuild-champion --out output_v3 || { log "MODEL REBUILD OR FINGERPRINT VERIFICATION FAILED - nothing published"; exit 2; }
fi

# 1b. one forecast per issue day: the first published forecast is canonical (no re-issue with later data)
pub="$STATE_DIR/public/data/predictions.json"
if [ -f "$pub" ] && [ "$($PY -c "import json;print(json.load(open('$pub'))['issue_time_utc'][:10])")" = "$today" ]; then
  log "today's forecast was already published; only resolving outcomes"
  $PY -W ignore -m fireaq prospective-log --state "$STATE_DIR" --no-predictions || exit 4
  cp "$STATE_DIR/prospective/summary.json" "$STATE_DIR/public/data/prospective_summary.json"; exit 0
fi

# 2. inference (writes $STATE_DIR/live/live_status.json whatever happens)
regions_arg=(); [ -n "${REGIONS:-}" ] && regions_arg=(--regions "$REGIONS")
$PY -W ignore -m fireaq predict-live --model output_v3/model_v3.joblib --live-dir "$STATE_DIR/live" \
    --predictions "$STATE_DIR/staging/predictions.json" ${regions_arg[@]+"${regions_arg[@]}"}
rc=$?
if [ ! -f "$STATE_DIR/live/live_status.json" ] || [ "$(find "$STATE_DIR/live/live_status.json" -mmin -120 | wc -l)" -eq 0 ]; then
  printf '{"state":"CRASH","checked_at_utc":"%s","reasons":["inference crashed before writing a status (exit %s); see the workflow log"]}\n' \
    "$(date -u +%FT%TZ)" "$rc" > "$STATE_DIR/live/live_status.json"
fi
state=$($PY -c "import json;print(json.load(open('$STATE_DIR/live/live_status.json'))['state'])" 2>/dev/null || echo CRASH)
log "inference exit=$rc state=$state"

# 3. validate before publishing: OK state, issued today, probabilities in [0,1], frozen model key
valid=0
if [ "$rc" -eq 0 ] && [ "$state" = "OK" ]; then
  $PY - "$STATE_DIR/staging/predictions.json" "$today" <<'PY' && valid=1
import json, sys, math
d = json.load(open(sys.argv[1])); frozen = json.load(open('release_v3/frozen_champion_v3.json'))
assert d['issue_time_utc'][:10] == sys.argv[2], 'issue date is not today'
assert d['model'] == frozen['champion']['key'], 'model key differs from the frozen champion'
assert d['predictions'], 'no predictions'
assert all(isinstance(p['probability_nextday_detection'], (int, float)) and math.isfinite(p['probability_nextday_detection'])
           and 0 <= p['probability_nextday_detection'] <= 1 for p in d['predictions']), 'invalid probability'
print('validated', len(d['predictions']), 'predictions')
PY
fi

# 4. prospective log: log published predictions; always try to resolve due outcomes
if [ "$valid" -eq 1 ] && [ "${DRY_RUN:-0}" != "1" ]; then
  $PY -W ignore -m fireaq prospective-log --state "$STATE_DIR" --predictions "$STATE_DIR/staging/predictions.json" || exit 4
else
  $PY -W ignore -m fireaq prospective-log --state "$STATE_DIR" --no-predictions || exit 4
fi

# 5. publish (static JSON for the website / API); predictions only when valid and not a dry run
cp "$STATE_DIR/live/live_status.json" "$STATE_DIR/public/data/live_status.json"
cp "$STATE_DIR/prospective/summary.json" "$STATE_DIR/public/data/prospective_summary.json"
$PY -W ignore -c "
import json; from fireaq import server
json.dump(server.regions(), open('$STATE_DIR/public/data/regions.json','w'), indent=1, default=str)
json.dump(server.model_card(), open('$STATE_DIR/public/data/model.json','w'), indent=1, default=str)" || true
cp web/index.html "$STATE_DIR/public/index.html"
if [ "$valid" -eq 1 ] && [ "${DRY_RUN:-0}" != "1" ]; then
  cp "$STATE_DIR/staging/predictions.json" "$STATE_DIR/public/data/predictions.json"; log "published"
else
  log "NOT publishing predictions (state=$state valid=$valid dry_run=${DRY_RUN:-0}); previous file stays and is flagged stale"
fi

# 6. keep state small: hashes of today's raw inputs are kept, raw bytes are not; NRT store trimmed to 60 days
[ -f "$STATE_DIR/live/manifest.json" ] && gzip -9 -c "$STATE_DIR/live/manifest.json" > "$STATE_DIR/manifests/$today.json.gz" && rm "$STATE_DIR/live/manifest.json"
rm -rf "$STATE_DIR/live/raw"
$PY -c "
import pandas as pd, pathlib
p = pathlib.Path('$STATE_DIR/live/nrt_store.parquet')
if p.exists():
    s = pd.read_parquet(p); cut = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=60)
    s[pd.to_datetime(s['acq_utc'], utc=True) >= cut].to_parquet(p, index=False)"

[ "$valid" -eq 1 ] && exit 0
[ "$state" = "CRASH" ] && exit 1
exit 3
