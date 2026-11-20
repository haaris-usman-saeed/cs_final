# Model card — FireRisk Open v0.3 (multi-region)

> **Research output only.** It is a calibrated probability of a *new NOAA-20 VIIRS satellite fire detection* in a 0.5° cell
> during the next UTC day. It is **not** an ignition, spread or danger probability. Do not use it for evacuation, suppression,
> dispatch or alerts.

Every number below is copied from `release_v3/RESULTS_v3.md` and `release_v3/test_results_v3.json`. Those files are
generated from the outputs, not typed by hand.

## Model
* **Champion:** `ExtraTreesClassifier(n_estimators=300, min_samples_leaf=50, random_state=17)` on feature set **B**.
  Set B is 32 features: season; NASA POWER (MERRA-2) weather and soil wetness lagged 2 days; 3–90-day rainfall and dry spell; ESA WorldCover
  fractions; Copernicus DEM elevation and roughness; neighbour/ring/regional NOAA-20 detections known in the last ≤ 7 days;
  and the cell's S-NPP 2012–2017 fire-day climatology. There are **no coordinates**.
  The output is isotonically calibrated on 2023, and the alert threshold is max-F1 on 2023 (0.1304 calibrated).
* **Selection** (pre-registered in `docs/PREREGISTRATION_v0.3.md`, committed before data and results):
  1. Compute pooled out-of-fold AP over rolling-origin folds (validating 2021, 2022, 2023) in 4 development regions.
  2. Keep every candidate within 1 bootstrap SE (0.0157) of the best.
  3. From those, pick the highest leave-one-region-out AP.
  4. Break ties on cross-fitted calibrated Brier.

  There were 13 candidates (3 dummy strategies, 2 climatologies and 8 learned families including XGBoost), 37 configurations, with every learned family fit on both feature set A and set B.
* **Frozen** at 2026-11-20 05:34 UTC. The frozen model files are hashed in `release_v3/frozen_model_sha256.txt` and committed
  (`ab35766`) **before** the one-time test was run.

## Fresh, untouched evaluation (run once)

| Test set | Cell-days | Positives | Base rate | Champion AP (95% CI, week blocks) | S-NPP climatology AP | Δ vs climatology (95% CI) |
|---|---|---|---|---|---|---|
| **T-time**: NRockies, Southwest, Southeast, 2024-02-02 → 2025-12-31 | 84,746 | 2,928 | 0.035 | **0.186** (0.155–0.221) | 0.121 | +0.046 to +0.086 |
| **T-space**: PNW + Ozarks (never used), 2024–25 | 49,761 | 3,084 | 0.062 | **0.130** (0.112–0.154) | 0.115 | +0.008 to +0.024 |
| T-space secondary: PNW + Ozarks, 2019–23 | 141,439 | 7,795 | 0.055 | 0.121 (0.107–0.135) | 0.109 | +0.007 to +0.016 |
| *NorCal 2024–25 (already evaluated in v0.2, **not untouched**)* | 31,039 | 1,119 | 0.036 | 0.094 (0.078–0.114) | 0.065 | +0.016 to +0.045 |

AP pooled across regions partly reflects differences in regional base rates. Within each region:

| Region (role) | Base rate | Champion AP (CI) | Climatology AP | Δ CI |
|---|---|---|---|---|
| NRockies (T-time) | 0.023 | 0.075 (0.060–0.090) | 0.052 | +0.014 to +0.033 |
| Southeast (T-time) | 0.151 | 0.264 (0.219–0.316) | 0.229 | +0.018 to +0.055 |
| Southwest (T-time) | 0.014 | 0.043 (0.035–0.055) | 0.032 | +0.005 to +0.018 |
| PNW (unseen, 2024–25) | 0.034 | 0.071 (0.060–0.084) | 0.063 | +0.003 to +0.015 |
| **Ozarks (unseen, 2024–25)** | 0.116 | 0.149 (0.126–0.181) | 0.142 | **−0.002 to +0.017 (not significant)** |

Other test metrics at the validation F1 threshold:

| Test set | ROC-AUC | Balanced acc. | Precision | Recall | Brier | Brier (base rate) |
|---|---|---|---|---|---|---|
| T-time | 0.840 | 0.683 | 0.204 | 0.426 | 0.0305 | 0.0334 |
| T-space | 0.726 | 0.634 | 0.137 | 0.458 | 0.0559 | 0.0589 |

**Event level.** On T-time, 1,080 events, the champion flags the first day of 31.6% of events while raising
72 alerts per 1,000 cell-days; climatology gets 31.9% with 101 alerts. At a matched alert rate (each model's top 5% on
2023) onset recall is 27.7% for the champion vs 22.4% for climatology. On T-space, 961 events, it is 28.3% vs 14.3%.

## Does fire history help? (matched Extra Trees, set A vs set B)

| | Δ AP (B − A), 95% CI, week blocks | Interpretation |
|---|---|---|
| Validation, development regions (OOF 2021–23) | −0.003 to +0.008 | no demonstrated gain |
| T-time (known regions, new years) | −0.002 to +0.008 | no demonstrated gain |
| **T-space (unseen regions), 2024–25** | **+0.011 to +0.027** | **gain** |
| T-space secondary, 2019–23 | +0.016 to +0.028 | gain |

Fire-history information, mostly the out-of-period S-NPP climatology, matters when the model must work **somewhere it
has never seen**. Within known regions, weather, soil, land cover and terrain already carry the same signal. The v0.2 finding
("fire features don't help") was partly an artefact of coordinates being features (`docs/AUDIT_v0.2.md` #2–#3).

## Comparison with v0.2
On NorCal 2024–25, v0.2 scored AP 0.085 and v0.3 scores 0.094. That set had already been inspected, so the comparison is
**indicative only**. The defensible improvements are the untouched tests above and the removal of coordinate memorisation.

## Limitations
* Skill is **modest**. In unseen regions precision is about 14% at 46% recall, and the margin over a simple fire climatology
  is small. In the Ozarks it is not significant.
* About 25–28% of positives continue a fire that started on the issue day after the cut-off. On rows with no issue-day detection,
  AP is 0.144 for T-time and 0.107 for T-space.
* Negatives are weak because clouds and smoke hide fires. Excluding cloud-obscured negatives (proxy) raises T-time AP to 0.239.
* Cells are 0.5° (≈ 2,000–2,500 km²). Agricultural and prescribed burning count as detections.
* The secondary continuous target (log1p detection count) has little skill: test R² is 0.057 at best (random forest, T-time) and ≤ 0.02
  in unseen regions. It is shown for completeness, not as a usable model.
* **Live inputs differ from training inputs.** Live uses NRT FIRMS without `type` (a static-source mask is applied instead) and
  POWER weather from **GEOS-IT** rather than MERRA-2 for recent days. The v0.2 vintage check measured how the fire and ERA5
  inputs differ. The MERRA-2→GEOS-IT shift is **not yet measured**.
* Supported area: only the 6 listed 4°×4° regions. The model refuses anywhere else.

## Intended use
Research, teaching, and benchmarking of leakage-safe spatio-temporal ML on open satellite data.
