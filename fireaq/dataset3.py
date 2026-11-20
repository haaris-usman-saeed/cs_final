"""Build the v0.3 released dataset (all regions) from data/v03/interim. Deterministic and idempotent.

Label statistics for untouched-test rows are NOT written to the coverage report during development
(``reveal_test_labels=False``); they are added only after the one-time test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .core import build_static_source_mask
from .experiment_v3 import assign_roles
from .features_v3 import DATASET_VERSION, SET_B, TARGET, make_dataset_v3
from .regions import REGIONS, STEP

DICTIONARY_V3 = {
    'region': 'Study region key (see fireaq/regions.py).', 'cell_key': 'region:cell_id, unique across regions.',
    'role': 'dev / test_time / test_space_primary / test_space_secondary / previously_tested_norcal / unused.',
    't2m_lag2': 'NASA POWER (MERRA-2) 2 m mean air temperature on UTC day d-2 (°C).',
    't2m_max_lag2': 'Daily maximum 2 m temperature on d-2 (°C).', 'rh2m_lag2': '2 m relative humidity, daily mean, d-2 (%).',
    'vpd_lag2': 'VPD from T2M_MAX and daily-mean RH2M (Tetens), d-2 (kPa); an upper-range approximation.',
    'ws10m_lag2': '10 m wind speed, daily mean, d-2 (m/s).', 'ws10m_max_lag2': '10 m wind speed, daily max, d-2 (m/s).',
    'rain_3d_lag2': 'PRECTOTCORR total over d-4..d-2 (mm).', 'rain_7d_lag2': 'over d-8..d-2 (mm).',
    'rain_30d_lag2': 'over d-31..d-2 (mm).', 'rain_90d_lag2': 'over d-91..d-2 (mm).',
    'dry_days_lag2': 'Consecutive days < 1 mm ending d-2, capped at 30.',
    'gwettop_lag2': 'MERRA-2 surface soil wetness (0-5 cm, fraction of saturation), d-2.',
    'gwetroot_lag2': 'MERRA-2 root-zone soil wetness (0-100 cm), d-2.',
    'gwetroot_chg30_lag2': 'Root-zone soil wetness change d-32 -> d-2 (drying is negative).',
    'lc_tree': 'ESA WorldCover 2020 tree-cover fraction of the cell.', 'lc_shrub': 'shrubland fraction.',
    'lc_grass': 'grassland fraction.', 'lc_crop': 'cropland fraction.', 'lc_built': 'built-up fraction.',
    'lc_bare': 'bare/sparse fraction.', 'lc_wetland': 'herbaceous wetland fraction.',
    'elev_mean_m': 'Copernicus DEM GLO-90 mean of 7x7 samples (m).', 'elev_std_m': 'std of the 49 samples (m), a roughness proxy.',
    'neighbor_fire_3d': 'Qualifying detections in the 8 neighbour cells known at issue, last 3 issue days.',
    'neighbor_fire_7d': 'Same, last 7 issue days.', 'neighbor_frp_3d': 'FRP (MW) of those, last 3 issue days.',
    'ring2_fire_7d': 'Detections in the 16 cells two steps away, last 7 issue days.',
    'region_mean_fire_3d': 'Mean known detections per land cell in the region, last 3 issue days.',
    'clim_fire_days_year': 'S-NPP 2012-2017 mean annual fire-days in the cell (pre-study; no leakage).',
    'clim_fire_days_month': 'S-NPP 2012-2017 mean fire-days in the cell in the issue calendar month.',
    'target_cloud_pct': 'POWER (CERES SYN1DEG) daily mean cloud amount on the target day (%). QC ONLY.',
    'target_cloud_obscured_proxy': 'target_cloud_pct >= 80. QC ONLY.',
    'weather_source_lag2': 'POWER source stream for the d-2 weather (MERRA2 historically; GEOSIT live).',
    TARGET: 'PRIMARY LABEL: 1 if >=1 qualifying NOAA-20 VIIRS detection (conf n/h, type 0) in the cell on UTC day d+1.',
    'target_nextday_count': 'Secondary continuous outcome: qualifying detections on d+1.',
    'in_primary_analysis': 'Land cell, eligible (no detection known in prior 7 days), eligibility certain, label ok.',
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_release_v3(root: Path = Path('data/v03'), reveal_test_labels: bool = False, log=print) -> dict:
    root = Path(root)
    interim, out = root / 'interim', root / 'dataset'
    out.mkdir(parents=True, exist_ok=True)
    status = pd.read_csv(interim / 'firms_day_status.csv')
    frames, masks = [], set()
    for r, spec in REGIONS.items():
        rd = interim / r
        cells = pd.read_csv(rd / 'cells.csv')
        firms = pd.read_parquet(rd / 'firms_regional.parquet')
        wx = pd.read_parquet(rd / 'weather.parquet')
        d = make_dataset_v3(firms, wx, cells, r, spec['bbox'], STEP, '2019-01-01', '2025-12-30',
                            day_status=status, keep_ineligible=True)
        frames.append(d)
        tr = firms[pd.to_datetime(firms['acq_date']).dt.year <= 2022]
        masks |= build_static_source_mask(tr)
        log(f'{r}: {len(d):,} cell-days')
    data = pd.concat(frames, ignore_index=True)
    data['in_primary_analysis'] = (data['eligible_new_activity'] & ~data['eligibility_uncertain']
                                   & (data['label_status'] == 'ok'))
    data['role'] = assign_roles(data).to_numpy()
    data['firms_product'] = 'NASA FIRMS NOAA-20 VIIRS 375m yearly US archive (viirs-jpss1), version 2 (SP)'
    data['weather_product'] = 'NASA POWER daily point API v2 (MERRA-2; clouds CERES SYN1DEG), UTC'
    data['static_products'] = 'ESA WorldCover 2020 v100; Copernicus DEM GLO-90; Natural Earth 10m; FIRMS S-NPP 2012-2017'
    data['dataset_version'] = DATASET_VERSION
    data['raw_manifest_sha256'] = _sha(root / 'manifest.json')
    data = data.sort_values(['region', 'issue_date', 'cell_key']).reset_index(drop=True)
    pq = out / f'firerisk_open_multiregion_v{DATASET_VERSION}.parquet'
    data.to_parquet(pq, index=False)
    (interim / 'static_source_mask_2019_2022_all_regions.json').write_text(json.dumps(sorted(masks)) + '\n')
    pd.DataFrame([{'column': c, 'dtype': str(data[c].dtype), 'description': DICTIONARY_V3.get(c, '')}
                  for c in data.columns]).to_csv(out / 'data_dictionary_v3.csv', index=False)
    prim = data[data['in_primary_analysis']]
    prim[prim.role == 'dev'].head(500).to_csv(out / 'preview_dev_500rows.csv', index=False)

    def label_stats(g):
        lab = g[TARGET]
        return {'rows': int(len(g)), 'positives': int(lab.sum()), 'prevalence': float(lab.mean()) if len(g) else None,
                'cloud_obscured_share': float(g['target_cloud_obscured_proxy'].mean()),
                'issue_day_detection_share_of_positives': float(g.loc[lab == 1, 'detection_on_issue_day'].mean())
                if lab.sum() else None}
    by = {}
    for (reg, role), g in prim.groupby(['region', 'role']):
        open_ = role == 'dev' or reveal_test_labels
        by.setdefault(reg, {})[role] = label_stats(g) if open_ else {'rows': int(len(g)),
                                                                       'labels': 'withheld until one-time test'}
    dev = prim[prim.role == 'dev']
    report = {'dataset_file': pq.name, 'sha256': _sha(pq), 'rows': int(len(data)), 'columns': int(data.shape[1]),
              'cells': int(data['cell_key'].nunique()), 'regions': sorted(data['region'].unique().tolist()),
              'in_primary_analysis': int(len(prim)), 'test_labels_revealed': reveal_test_labels,
              'primary_by_region_and_role': by,
              'dev_by_target_year': {int(k): label_stats(g) for k, g in
                                     dev.groupby(pd.to_datetime(dev['target_date']).dt.year)},
              'row_accounting': {
                  'not_eligible_recent_detection': int((~data['eligible_new_activity']).sum()),
                  'eligibility_uncertain_firms_day': int((data['eligible_new_activity'] & data['eligibility_uncertain']).sum()),
                  'label_withheld_firms_target_day': int((data['eligible_new_activity'] & ~data['eligibility_uncertain']
                                                          & (data['label_status'] != 'ok')).sum())},
              'feature_missing_rows_dropped_note': 'rows with any missing feature are not in the file (no imputation)',
              'static_source_mask_boxes': len(masks)}
    (out / 'coverage_report_v3.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    (out / 'SHA256SUMS').write_text(''.join(f'{_sha(p)}  {p.name}\n' for p in sorted(out.iterdir())
                                            if p.is_file() and p.name != 'SHA256SUMS'))
    log(f"v0.3 dataset: {len(data):,} cell-days, {len(prim):,} primary; dev positives {int(dev[TARGET].sum()):,}")
    return report
