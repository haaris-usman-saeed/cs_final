# FireRisk Open v0.3

> **Research prototype. Not for evacuation, suppression, dispatch, public alerts or any emergency decision.**
> Outputs are calibrated probabilities of a *new NOAA-20 VIIRS satellite active-fire detection* in a 0.5° cell during the
> next UTC day — **not** probabilities of wildfire ignition, spread or danger. A non-detection is not evidence of no fire.

An MIT-licensed, reproducible pipeline and **original open dataset** linking NASA FIRMS satellite fire detections with
NASA POWER weather and soil moisture, ESA WorldCover land cover, Copernicus DEM terrain and a pre-2018 fire climatology,
for **6 US regions** (2019–2025). It includes a leakage-safe model comparison evaluated once on regions and years never
used for development, and a read-only prediction website/API fed by a fail-safe daily NRT job.

**Task.** At 00:00 UTC on day *d*, for each land cell with no detection known in the previous 7 days: will NOAA-20 record a
qualifying (nominal/high-confidence, vegetation-type) detection in that cell during UTC day *d+1*?

## Results in one table (one-time untouched test; details in `release_v3/RESULTS_v3.md`)

| Test | Base rate | **Model AP** (95% CI) | Fire-climatology AP | Gain over climatology |
|---|---|---|---|---|
| New years 2024–25 in NRockies / Southwest / Southeast | 0.035 | **0.186** (0.155–0.221) | 0.121 | clear (CI +0.046 to +0.086) |
| **Never-used regions** PNW + Ozarks, 2024–25 | 0.062 | **0.130** (0.112–0.154) | 0.115 | small (CI +0.008 to +0.024); Ozarks alone not significant |

* Fire-history features: **no demonstrated gain inside known regions; a significant gain in unseen regions** (Δ AP CI +0.011 to +0.027).
* Skill is modest: in unseen regions about 14% of alerts are followed by a detection and 46% of detections were alerted.
* v0.2's single-region model scored AP 0.085 on NorCal 2024–25; v0.3 scores 0.094 there, but that set was already seen, so treat it as indicative only.

Read `docs/MODEL_CARD_v0.3.md`, `docs/DATA_CARD_v0.3.md`, `docs/AUDIT_v0.2.md` and `docs/PREREGISTRATION_v0.3.md`.

## Use the data (no downloads)

```python
import pandas as pd
df = pd.read_parquet('release_v3/firerisk_open_multiregion_v0.3.0.parquet')
dev = df[df.in_primary_analysis & (df.role == 'dev')]        # development rows
print(dev.groupby('region').target_nextday_detection.mean())
```

## Reproduce from a clean checkout

Python ≥ 3.10. The reported run used 3.14.5 on macOS arm64; exact versions are in `requirements-lock.txt`.

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

On macOS, if `import xgboost` fails for lack of `libomp`, this script fixes it inside the venv only (or `brew install libomp`):

```bash
scripts/macos_xgboost_libomp.sh
```

```bash
python -m pytest -q
```

Keyless ingest, about 15–25 min. It downloads ~555 MB of FIRMS archives, 381 POWER requests and the WorldCover/DEM COG windows, and resumes from cache:

```bash
python -m fireaq ingest-v3
```

Deterministic dataset build. Test-label statistics are withheld unless you add `--reveal-test-labels`:

```bash
python -m fireaq build-v3
```

Development stage, about 20 min. It uses development rows only, then freezes the champion to `output_v3/`:

```bash
python -m fireaq experiment-v3
```

One-time untouched test. It refuses to run twice or if the development rows changed:

```bash
python -m fireaq evaluate-test-v3
```

```bash
python scripts/make_report_v3.py
```

## Live predictions, website and API

```bash
python -m fireaq predict-live
```

The daily job does inference only, never retraining. It:
1. checks the health of all FIRMS NRT feeds and uses only NOAA-20, the training sensor;
2. appends to a local NRT store with first-seen times, so latency can be measured;
3. refuses to publish when the feed is empty or stale (> 18 h), NRT history doesn't cover the 7-day window, a day in that window has a detection gap,
   or POWER weather for d−2 is missing for > 10% of cells;
4. scores the 6 regions with the frozen model.

Every run writes `data/live/live_status.json`. Schedule it daily with `scripts/update_daily.sh`; no job is installed by default.

```bash
uvicorn fireaq.server:app --host 127.0.0.1 --port 8000
```

```bash
curl -s "http://127.0.0.1:8000/api/predictions?region=pnw"
```

| Endpoint | Returns |
|---|---|
| `/` | Website: region tabs, grid, per-region untouched-test skill, data freshness, refusal reasons |
| `/api/predictions?region=&cell_id=` | Latest predictions, freshness, `stale` flag (not issued today, or > 36 h old), uncertainty, unscored cells with reasons; **503** if none |
| `/api/model` | Champion, selection rule, validation ablation, all test sets with CIs and event metrics |
| `/api/regions` | Supported regions; nothing outside them is predicted |
| `/api/status` | Last live-run state and checks |

Live inputs differ from training inputs: NRT detections have no `type` field, and POWER weather is GEOS-IT after
2026-10-02. The weather shift was measured with `python -m fireaq vintage-check-v3`:
- RH −3.9 points, wind −0.7 m/s, temperature +0.7 °C (region-dependent);
- under these shifts, 99.4% of the frozen model's alert decisions are unchanged.

The FIRMS NRT-vs-archive difference is not measured. Treat live output as experimental.

## Running it as a platform
* **CI** (`.github/workflows/ci.yml`) runs on every push and PR:
  * tests on Python 3.11 and 3.12;
  * a frozen-model rebuild and fingerprint check;
  * a live dry run on real feeds that never publishes.
* **Daily job** (`.github/workflows/daily.yml` → `scripts/daily_job.sh`, 02:30 UTC, default branch only):
  * rebuilds and verifies the frozen model;
  * runs inference with every safeguard;
  * publishes only on success, and appends to the **prospective log**;
  * opens a `live-run-failure` issue whenever nothing is published.
* **Prospective log** (`state/prospective/log.jsonl` on the `prospective-state` branch): append-only and hash-chained.
  Each prediction records its model fingerprint, issue time and input vintage; observed outcomes are appended later.
  `python -m fireaq prospective-report` summarises it. **No live accuracy is reported** until 60 resolved issue days and
  50 observed detections exist.
* **NRT vs archived FIRMS** (`python -m fireaq nrt-vs-sp`): this compares coverage, timing, confidence, geolocation and label
  agreement. It is **not yet measured**, because it needs a FIRMS MAP_KEY or the 2026 archive.
* Costs, requirements, licences and the clean-checkout reproduction record are in **`docs/OPERATIONS.md`**.

## Earlier pilot (v0.2, NorCal only)
Kept for reproducibility: `ingest-archive`, `build`, `experiment`, `evaluate-test`, `vintage-check` and `predict-live-v02`, with
results in `release/RESULTS.md` and `docs/MODEL_CARD.md`.

## Attribution and licences
NASA FIRMS (LANCE / ESDIS) · NASA POWER (NASA LaRC; MERRA-2, GEOS-IT, CERES) · ESA WorldCover 2020 (CC BY 4.0) · Copernicus DEM
GLO-90 (© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the EU and ESA) ·
ECMWF ERA5 via Open-Meteo (v0.2 only, CC BY 4.0) · Natural Earth (public domain). No endorsement implied.
Code: MIT. Datasets: CC BY 4.0. Upstream terms are not superseded.
