# Model card — FireRisk Open v0.2 pilot champion

> **Research hindcast only. Not for evacuation, suppression, dispatch, alerts or any emergency decision.**
> The output is the probability of a NOAA-20 VIIRS *satellite detection*, not of an ignition, spread or danger.

All numbers below come from `release/RESULTS.md`, `release/test_results.json` and `release/validation_leaderboard.csv`.

## Model
* **Champion:** `ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_leaf=20, random_state=17)`
  on the `weather_plus_fire` feature set (21 features: location, season, ERA5 weather lagged 2 days, and fire history
  known at issue). Isotonic calibration and the decision threshold (0.0803) were fit on validation only.
* **Selection:** the pre-registered rule, highest validation average precision across 11 model families × 2 feature
  sets × bounded grids (72 configurations). Frozen 2026-11-20T04:25Z; test run once afterwards.
* **Region and horizon:** 64 cells of 0.5° in bbox −123.5, 38.5, −119.5, 42.5. Issue 00:00 UTC, target UTC day d+1.
  `predict_prob` refuses coordinates outside this region.

## Data
| Split | Target dates | Rows | Positives | Prevalence | Distinct positive events |
|---|---|---|---|---|---|
| train | 2019-01-02 → 2022-12-31 | 66,584 | 2,312 | 3.47% | 982 |
| validation | 2023-02-02 → 2023-12-31 | 13,824 | 486 | 3.52% | 195 |
| test | 2024-02-02 → 2025-12-31 | 31,039 | 1,119 | 3.61% | 427 |
| embargo (excluded) | Jan 2023, Jan 2024 | 3,390 | 125 | — | — |

The 32-day embargo comes before validation and before test. Pairwise key overlap between splits is 0 (`release/split_manifest.json`).
"Events" are positive cell-days linked by spatial adjacency within 2 days.

## Performance (test, evaluated once)

| | AP | ROC-AUC | Brier | Balanced acc. | Recall | Precision |
|---|---|---|---|---|---|---|
| Base rate (no skill) | 0.036 | 0.500 | — | 0.500 | — | — |
| Climatology (cell × month, calibrated) | 0.076 | 0.675 | 0.0346 | 0.583 | 0.228 | 0.122 |
| Matched weather-only Extra Trees | 0.086 | 0.711 | 0.0338 | 0.617 | 0.335 | 0.110 |
| **Champion** | **0.085** | **0.705** | **0.0339** | **0.636** | **0.457** | **0.084** |

* 95% CI for champion AP (week-cluster bootstrap): 0.071–0.102.
* Δ AP vs climatology: CI −0.003 to +0.020 by week clusters, which **includes zero**; +0.0003 to +0.016 by cell clusters.
* Δ AP vs weather-only: CI −0.007 to +0.006. **Fire-history features add no demonstrable skill.**
* Accuracy is 0.80 vs 0.96 for always-"no". Accuracy is not a meaningful metric here.

## Honest assessment
* Skill over a simple seasonal/geographic climatology is **small and not robust to time-clustered uncertainty**.
  The model ranks cells about 2.4× better than chance, but at the chosen threshold ~92% of alarms are false.
* About 25% of positives are probably continuations: the fire started on the issue day after the 00:00 cut-off.
  These are the "easy" positives (AP 0.33). For genuinely new activity, AP is 0.064 vs 0.057 for climatology.
* **Spatial transfer:** retraining without each 2°×2° block lowers AP in every block (NE 0.037→0.029,
  NW 0.060→0.058, SE 0.076→0.066, SW 0.134→0.097). Do not extend to other regions without new validation.
* Negatives are weak. A sensitivity analysis dropping negatives flagged as cloud-obscured raises AP from 0.085 to 0.108,
  so label noise from obscuration is material.
* Most important features (validation permutation AP drop): latitude, own-cell 8–30-day fire history, longitude,
  season. Much of the signal is "where and when fires usually are".

## Live use
Live inputs differ from training: NRT detections without `type` (a static-source mask is used instead) and
operational forecast-model weather instead of ERA5. `release/vintage_check.json` quantifies part of this, using the
frozen model after the test:

* **Weather vintage.** This compares Open-Meteo archived operational forecasts with ERA5 over 16 cells and 11,200 cell-days
  in 2024–25. Biases: RH −7.8 points, min RH −2.2, wind +0.86 m/s, precipitation −0.77 mm/day, temperature +0.08 °C
  (r = 0.97). On 7,529 paired test rows, AP falls from 0.079 to 0.075, mean |Δp| is 0.007, and 93.5% of threshold
  decisions are unchanged.
* **Fire `type` field.** Re-deriving test labels and features without `type`, with the 2019–2022 static-source mask
  instead, keeps 3,732 shared positive cell-days, loses 45 and adds 116 (out of 44,480), with 532 eligibility flips.
  AP is 0.0857 vs 0.0854.
* **Not measured:** standard-processing vs NRT detection and geolocation differences, and real NRT latency. Both need a
  MAP_KEY NRT archive or prospective collection.

Live output is
labelled **EXPERIMENTAL – UNVALIDATED UNDER LIVE INPUTS** and is withheld when NRT history does not cover the
30-day fire window. Daily inference re-scores with the frozen model. Retraining needs newly published,
verified standard-processing labels plus a fresh pre-registered validation/test cycle.

## Recommended next steps
Use the VIIRS Level-2 cloud/fire mask (VJ114) for observation opportunity, add fuel/land-cover features
(e.g. LANDFIRE), separate agricultural burning, train on archived operational weather, and run a prospective
evaluation of the live pipeline.
