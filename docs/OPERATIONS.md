# Operating FireRisk Open — deployment, costs, licences, reproducibility

Status (2026-11-20): **experimental research platform**. Live predictions are satellite-detection probabilities, not
wildfire-risk or ignition forecasts. **No live accuracy is claimed** until the prospective log
(`state/prospective/log.jsonl`) has at least 60 resolved issue days and 50 observed detections.

## 1. What runs where

| Component | How | Trigger |
|---|---|---|
| Tests (38, synthetic data only) | `.github/workflows/ci.yml` → `tests`, Python 3.11 and 3.12 | every push to `main` / `dev/**`, every PR |
| Frozen-model reproducibility | CI job `frozen-model-reproducibility`: rebuilds the v0.3 champion from the committed dataset and checks the prediction fingerprint (tolerance 1e-9) | every push / PR |
| Live dry run | CI job `live-dry-run`: real feeds, NorCal only, publishes nothing; a safeguard refusal (exit 3) is accepted | PRs, manual |
| **Daily inference** | `.github/workflows/daily.yml` → `scripts/daily_job.sh` | 02:30 UTC daily, **only from the default branch** (it does not run until this PR is merged) |
| Website | Either `uvicorn fireaq.server:app` (FastAPI) or a static site from the `prospective-state` branch `public/` folder (GitHub Pages; off unless the repo variable `PUBLISH_PAGES=true`) | — |

### Daily job safeguards (`scripts/daily_job.sh`)
1. **Model integrity.** The 107 MB frozen model is not committed. Each runner rebuilds it from
   `release_v3/firerisk_open_multiregion_v0.3.0.parquet` and verifies it against `release_v3/model_fingerprint_v3.json`
   (2,000 reference rows). Any mismatch exits with code 2 and publishes nothing. Measured: max |Δp| ≤ 1.5e-14 on macOS, and identical 9-decimal hashes.
2. **Input checks** (`fireaq/live_v3.py`). The run is refused (exit 3, nothing published) if:
   * the NOAA-20 NRT feed is empty or its newest detection is more than 18 h old;
   * the NRT history does not cover the 7-day window;
   * any full day in that window has an NRT gap (no detections, a missing day or night overpass, or < 10% of the median count);
   * POWER d−2 weather is missing for more than 10% of cells;
   * the static-source mask is missing.
3. **Output validation.** The issue date must be today, the model key must equal the frozen champion, and all probabilities must be finite and in [0, 1].
4. **One forecast per issue day.** The first published forecast is canonical; reruns that day only resolve outcomes.
5. **Status is always published** (`public/data/live_status.json`). An old `predictions.json` is never deleted, but the
   API and website mark it **STALE** whenever its issue date is not today.
6. **Prospective log** (`fireaq/prospective.py`):
   * it is append-only and hash-chained, and a conflicting re-log of a past prediction is refused;
   * the workflow runs `scripts/check_append_only.py` before committing, so any edit to an earlier line fails the run;
   * outcomes are resolved from NRT only after the target day plus 12 h and only with full coverage, otherwise they are logged as `unresolvable_gap`.
7. **Alerts.** A non-zero job exit:
   * opens or comments on a `live-run-failure` GitHub issue;
   * fails the workflow, so GitHub e-mails the workflow owner;
   * uploads the logs as an artifact.

Exit codes: `0` published · `2` model rebuild/verification failed · `3` safeguard refusal · `4` prospective-log error · `1` crash.

### Verified locally on 2026-11-20 (real feeds)
* **Fresh state at 18:47 UTC:** `REFUSED_HISTORY`. The keyless 7-day feed started at Nov 13 00:15, but the window needed
  Nov 12 21:00. Exit 3, status published, no predictions. This is why the schedule is 02:30 UTC.
* **State seeded with an existing NRT store:** OK, 196 cells scored, validated, published, and 196 prediction records logged.
  A second run the same day did not re-issue. The append-only check passed.

## 2. Requirements
* Python 3.11–3.14 (CI covers 3.11 and 3.12; the reported runs used 3.14.5 on macOS arm64). Dependencies are in `pyproject.toml`, and exact
  versions in `requirements-lock.txt`.
* macOS only: the XGBoost wheels need `libomp`. Run `scripts/macos_xgboost_libomp.sh` (venv-local) or `brew install libomp`.
  XGBoost is only used for model comparison, not by the deployed Extra Trees champion.
* Memory: the model rebuild peaks at **3.6 GB RSS** (measured). Daily inference and the API need much less.
  The full development stage needs more; 8 GB+ is recommended.
* Network:
  * daily inference needs `firms.modaps.eosdis.nasa.gov` and `power.larc.nasa.gov`;
  * the full rebuild additionally needs `esa-worldcover.s3.eu-central-1.amazonaws.com`, `copernicus-dem-90m.s3.amazonaws.com` and `raw.githubusercontent.com`.
  * **No API keys are required.** `FIRMS_MAP_KEY` is optional (NRT backfill; NRT-vs-SP comparison).

## 3. Operating costs (measured where stated)

| Item | Amount | Cost |
|---|---|---|
| GitHub Actions, daily job | ~10 min/run, estimated from local measurements (inference ~8 min on a laptop, rebuild ~8 s) plus ~1–2 min pip install; not yet measured on a GitHub runner → ~300 runner-min/month | **free**: the repository is public, and standard runners are free for public repos. A private repo on the Free plan has 2,000 min/month |
| GitHub Actions, CI | ~4 jobs × 3–8 min per push/PR | free (public repo) |
| State branch growth | prospective log ~170 KB/day (196 predictions, measured), gzipped input manifests ~29 KB/day, public JSON ~165 KB overwritten daily | ≈ 70–130 MB/year before git compression; consider yearly log rotation (start a new chained file) |
| Actions cache | NRT store ~0.4 MB per 7 days, capped at 60 days (~3.5 MB) | free (10 GB per repo) |
| Upstream API load per day | FIRMS: one ~1 MB CSV plus 4 HEAD requests; NASA POWER: one request per land cell (381) | free, no keys; stay within fair use and don't run more often than daily |
| Website | GitHub Pages static site, or any small VM running uvicorn | Pages free for public repos; a VM is optional |
| Full rebuild from scratch | ~555 MB of FIRMS archives + 381 POWER calls + WorldCover/DEM windows; ingest ~15 min (measured, clean clone) | free |

## 4. Licences and attribution (all sources keyless)

| Source | Used for | Terms |
|---|---|---|
| NASA FIRMS (LANCE / ESDIS), NOAA-20 & S-NPP VIIRS 375 m | labels, fire features, live NRT | NASA open data; cite FIRMS and do not imply endorsement |
| NASA POWER (MERRA-2, GEOS-IT, CERES SYN1DEG) | weather and soil wetness | NASA open data; cite the POWER Project (NASA Langley) |
| ESA WorldCover 2020 v100 | land-cover fractions | **CC BY 4.0**: © ESA WorldCover project / contains modified Copernicus Sentinel data (2020) |
| Copernicus DEM GLO-90 | terrain | free with attribution: © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the EU and ESA |
| Natural Earth 10 m | land/lake mask | public domain |
| ECMWF ERA5 via Open-Meteo | v0.2 dataset; vintage checks only | CC BY 4.0 (Open-Meteo); free API for non-commercial use |
| **This repository** | code | MIT |
| **Released datasets** (`release/`, `release_v3/`) | derived data | CC BY 4.0, with the attributions above kept |

## 5. Can someone reproduce the full pipeline from a clean checkout?
**Tested on 2026-11-20.** I made a fresh `git clone` of `dev/v0.3-multiregion` (commit `3b97017`), set up a new venv, and used no cached data
(log: `docs/REPRODUCIBILITY_RUN_2026-11-20.md`).

| Step | Result |
|---|---|
| `pip install -e ".[dev]"` + macOS libomp fix | OK |
| `pytest` | 34/34 passed (the suite at that commit) |
| `ingest-v3` (409 downloads, no keys) | OK, ~15 min |
| `build-v3` | **Dataset content identical to the released file** (973,836 × 65). Only the `raw_manifest_sha256` column differs, because retrieval timestamps differ |
| `experiment-v3`, `evaluate-test-v3` | see `docs/REPRODUCIBILITY_RUN_2026-11-20.md` |

Known reproducibility limits:
* Upstream sources can revise data. FIRMS replaces NRT with SP, and POWER and WorldCover releases change, so a future
  rebuild may differ. The SHA-256 manifests show exactly what was used.
* The test set has already been evaluated, so re-running `evaluate-test-v3` reproduces numbers but is not a new test.
* The NRT-vs-SP comparison is **not yet measured**. It needs a FIRMS MAP_KEY (`python -m fireaq nrt-vs-sp --mode api`), an Earthdata
  token for LANCE NRT archives, or the 2026 SP yearly archive (not yet published; check with `--mode store`).
