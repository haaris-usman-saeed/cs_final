# Methodological audit of v0.2 (written 2026-11-20, before any v0.3 data or results)

Scope: `fireaq/` at commit `74b3fe0`, `release/` outputs, and the docs. Severity: **H** = can change conclusions,
**M** = biases estimates or limits use, **L** = presentation or robustness.

| # | Sev | Finding | Evidence | v0.3 response |
|---|---|---|---|---|
| 1 | H | **Champion chosen on one validation year.** 2023 has 486 positives in 195 events. The top three validation APs (0.0876 / 0.0853 / 0.0830) are within sampling noise, so "champion" is mostly luck. | `release/validation_leaderboard.csv` | Rolling-origin CV over 3 validation years × 4 regions (pooled out-of-fold AP), with a pre-registered one-standard-error rule |
| 2 | H | **`latitude`/`longitude` are features.** In a single 4°×4° region they let trees memorise *where* fires happen. That explains the large spatial hold-out drop (SW 0.134→0.097) and makes the model non-transferable. | permutation importance ranks lat #1, lon #3 | Coordinates removed. Location enters only through physical static layers (land cover, terrain) and an out-of-period fire climatology |
| 3 | H | **The fire-history ablation was confounded.** The "weather-only" arm kept lat/lon + day-of-year, which already encode the regional fire climatology. Recent fire history was therefore redundant *by construction*, so "fire features don't help" was not a clean test. | feature lists in `core.py` v0.2 | Arms: **A** = weather + soil + land cover + terrain + season (no fire information of any kind, no coordinates); **B** = A + fire history. Same rows, families, grids and folds |
| 4 | H | **Climatology baseline is not available for new regions.** The cell×month rate is target-encoded from training labels, so in an unseen region it collapses to a month-only rate. | `Climatology` fallback | Add a historical climatology from the S-NPP archive 2012–2017 (before the study period, available for any region) as baseline and feature |
| 5 | M | **The 2024–25 test has been used.** It cannot serve as an untouched test for any later version. | `release/test_results.json` | Fresh test = 2 never-inspected regions (all years) + 2024–25 in 3 new development regions. NorCal 2024–25 is reported only as "previously evaluated" |
| 6 | M | **Threshold and calibration were fit on the same validation predictions used for selection.** The in-sample threshold is optimistic. | `run_development` | Cross-fitted calibration/threshold metrics during selection. The final calibrator is fit on the embargoed 2023 fold only |
| 7 | M | **No event-level evaluation.** About 25% of positives are continuations of fires that started on the issue day. Cell-day AP mixes "new event onset" with "next day of a known event". | `coverage_report.json` | Event-level onset recall, event recall and false-alarm rate, plus metrics on rows with no issue-day detection |
| 8 | M | **No fuel, vegetation, soil-moisture or terrain information.** Fire-weather alone cannot tell forest from cropland, which is the main driver of agricultural burning in the Sacramento Valley. | feature list | ESA WorldCover 2020 class fractions, Copernicus DEM elevation/roughness, MERRA-2 surface/root-zone soil wetness and 90-day dryness |
| 9 | M | **Training weather (ERA5) ≠ live weather (forecast models).** Measured: RH −7.8 pts, wind +0.86 m/s. | `vintage_check.json` | NASA POWER for training and live (MERRA-2 history, GEOS-IT near-real-time; same API and variables). The remaining MERRA-2 vs GEOS-IT shift is documented |
| 10 | M | **The live model needed 30 days of NRT history** (`own_fire_30d`), so keyless live scoring was impossible for 30 days. | `live_status.json` | Fire-history set B is pre-registered as **live-computable**: recent windows ≤ 7 days + static climatology |
| 11 | M | **`region_fire_3d` depends on bbox size** (a sum over 64 cells), so it is not comparable across regions. | `core.py` | Replaced by the regional per-cell mean |
| 12 | L | **Observation mask is only a weather-model cloud proxy** at overpass hours. | DATA_CARD | Still a proxy (L2 VIIRS cloud masks need Earthdata login). Reported as a sensitivity analysis. National day-status QC kept |
| 13 | L | **Brier of uncalibrated, class-weighted models appeared in the leaderboard** next to calibrated ones. | leaderboard | Only cross-fitted calibrated Brier is reported for selection |
| 14 | L | **Single region, single fire regime.** Generalisation claims were impossible. | — | 6 regions with different regimes |

Things that were checked and found sound: target-date splits with a 32-day embargo, an availability-time cut-off
for fire features (3 h latency), withholding labels on missing FIRMS days, deterministic dataset build, one-time test
guard, and no imputation of missing weather.
