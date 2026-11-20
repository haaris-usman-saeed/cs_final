# FireRisk Open v0.2

> **Research prototype. Not for evacuation, suppression, dispatch, public alerts or any emergency decision.**
> It estimates the probability of a NOAA-20 VIIRS *satellite active-fire detection*, not wildfire ignition,
> spread or danger. A non-detection is not evidence that there was no fire.

An MIT-licensed, reproducible pipeline and **original dataset** linking NASA FIRMS VIIRS active-fire detections
with ECMWF ERA5 weather on a 0.5° grid. It includes a leakage-safe model comparison and a read-only prediction API.

**Task.** For a land cell at 00:00 UTC on issue day *d*, among cells with no detection *known at issue time*
during the previous 7 days, predict whether a qualifying NOAA-20 VIIRS detection (nominal/high confidence,
presumed vegetation fire) occurs in the cell during UTC day *d+1* (24–48 h ahead).

## Status (measured, 2026-11-20)

| Component | Status |
|---|---|
| Real data | ✅ 8 NOAA-20 FIRMS yearly archives (2018–2025) + 512 ERA5 chunks + Natural Earth, all SHA-256 manifested (`release/ingest_manifest.json`) |
| Dataset | ✅ `release/firerisk_open_norcal_v0.2.0.parquet`: 163,584 cell-days, 114,837 in primary analysis, 4,042 positives. Rebuild is byte-identical |
| Models | ✅ 11 families × 2 feature sets, validation-only selection, one-time test (`release/RESULTS.md`) |
| Skill | ⚠️ **Weak.** Test AP 0.085 vs climatology 0.076 vs base rate 0.036. The gain over climatology is not robust to week-clustered bootstrap (CI includes 0). Fire-history features add nothing over weather-only |
| Spatial transfer | ⚠️ AP drops in all 4 held-out 2°×2° blocks |
| Live predictions | ⚠️ Pipeline works keylessly, but publishing is **blocked** until the local NRT store spans 30 days, or until a `FIRMS_MAP_KEY` is set to backfill. Live inputs are unvalidated |
| FIRMS MAP_KEY API path | ⏸ Implemented, not exercised (no key available) |
| PM2.5 module | ⛔ Blocked: OpenAQ v3 requires an API key (HTTP 401) |

Read `docs/MODEL_CARD.md` and `docs/DATA_CARD.md` before using anything here.

## Use the released data (no downloads needed)

```python
import pandas as pd
df = pd.read_parquet('release/firerisk_open_norcal_v0.2.0.parquet')
primary = df[df.in_primary_analysis]          # modelling population
print(primary.target_nextday_detection.mean())
```

Column definitions are in `release/data_dictionary.csv`. Checksums are in `release/SHA256SUMS`.

## Reproduce everything from a clean checkout

Python ≥ 3.10 (the reported run used 3.14.5 on macOS arm64; exact versions are in `requirements-lock.txt`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```

```bash
python -m fireaq ingest-archive --out data/norcal
```

That command downloads about 345 MB from FIRMS plus 512 Open-Meteo requests, roughly 1–2 h because of the fair-use pacing. It needs no key. It resumes from cache if interrupted.

```bash
python -m fireaq build --input data/norcal
```

```bash
python -m fireaq experiment --data data/norcal/dataset/firerisk_open_norcal_v0.2.0.parquet --out output
```

```bash
python -m fireaq evaluate-test --data data/norcal/dataset/firerisk_open_norcal_v0.2.0.parquet --out output
```

```bash
python -m fireaq vintage-check --data data/norcal/dataset/firerisk_open_norcal_v0.2.0.parquet --out output
```

```bash
python scripts/make_report.py
```

`experiment` never reads test labels. It writes the validation leaderboard and `output/frozen_champion.json`.
`evaluate-test` refuses to run a second time in the same output directory, and refuses if the dataset hash changed.
The design was fixed in `docs/PREREGISTRATION.md` before any metric was seen.

## Live predictions and the API

```bash
python -m fireaq predict-live
```

`predict-live` downloads the keyless NOAA-20 NRT 7-day feed into `data/live/nrt_store.parquet`, pulls Open-Meteo
forecast-API weather, builds the 00:00 UTC snapshot and scores it with the frozen model.
It **refuses** to publish if the NRT history does not cover the champion's 30-day fire window. In that case it writes
`data/live/live_status.json` with state `BLOCKED_INSUFFICIENT_NRT_HISTORY`. To unblock, either
`export FIRMS_MAP_KEY=...` (free from https://firms.modaps.eosdis.nasa.gov/api/map_key/; backfilled automatically)
or run it daily (e.g. `scripts/update_daily.sh`) until the store spans 31 days.

```bash
uvicorn fireaq.server:app --host 127.0.0.1 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/api/model
```

```bash
curl -s "http://127.0.0.1:8000/api/predictions?cell_id=r4_c3"
```

| Endpoint | Returns |
|---|---|
| `/` | Dashboard: grid, live status, frozen-model metrics |
| `/api/predictions` | Latest predictions, `age_hours`, `stale` flag; **503** if none exist; `?cell_id=` filter |
| `/api/model` | Champion, split, test metrics with bootstrap CIs |
| `/api/status` | Last live run state (e.g. blocked reason) |
| `/health` | Liveness and whether predictions/metrics exist |

The server never calls data sources or models on request, and never shows placeholder numbers.
**Daily inference** re-scores with the frozen model. **Retraining** needs newly published standard-processing labels
and a new pre-registered validation/test cycle.

## Repository layout

```
fireaq/        sources.py (clients) · ingest.py · geo.py (land mask, cell area) · core.py (labels/features)
               dataset.py (release) · experiment.py (models, splits, test-once) · vintage.py · model.py · server.py · cli.py
docs/          PREREGISTRATION.md · DATA_CARD.md · MODEL_CARD.md · DECISIONS.md · CLAUDE_HANDOFF.md
release/       dataset, data dictionary, coverage/ingest reports, manifests, leaderboards, test results, RESULTS.md
tests/         synthetic-only unit/integration tests
legacy/        original PM2.5 feasibility probe (reference only)
data/, output/ git-ignored working directories (raw/ → interim/ → dataset/)
```

## Data sources, licences, attribution

* **NASA FIRMS** NOAA-20 VIIRS 375 m active fire (yearly country archive; NRT feed). "We acknowledge the use of data
  and imagery from LANCE FIRMS operated by NASA's Earth Science Data and Information System (ESDIS) with funding
  provided by NASA Headquarters." NASA data are openly available; no endorsement implied.
* **ECMWF ERA5** (Hersbach et al., 2020) and operational forecast models via **Open-Meteo** (CC BY 4.0, attribution required;
  the free API is for non-commercial use).
* **Natural Earth** land/lakes (public domain).
* Released dataset: CC BY 4.0. Code: MIT. Upstream terms are not superseded by this repository's licences.

See `CHANGELOG.md` for audit findings and fixes.
