"""Build the released dataset layer from an archive ingest directory (deterministic, idempotent)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import DATASET_VERSION, FEATURES, Grid, build_static_source_mask, make_dataset
from .ingest import NEIGHBOR_PAD_CELLS

DICTIONARY = {
    'cell_id': 'Row/column id within the study bbox (r{row}_c{col}, row 0 = southernmost).',
    'cell_global_id': 'Bbox-independent id g{step}_y{floor((lat+90)/step)}_x{floor((lon+180)/step)}.',
    'issue_date': 'UTC calendar date d; the forecast is issued at d 00:00 UTC.',
    'target_date': 'UTC date d+1 whose detections define the label.',
    'issue_time_utc': 'ISO timestamp of issue (d 00:00Z).',
    'target_start_utc': 'Start of label window (d+1 00:00Z), inclusive.',
    'target_end_utc': 'End of label window (d+2 00:00Z), exclusive.',
    'latitude': 'Cell centre latitude (deg, WGS84).', 'longitude': 'Cell centre longitude (deg).',
    'land_fraction': 'Natural Earth 10 m land-minus-lakes fraction of the cell (area weighted lattice).',
    'cell_area_km2': 'Spherical area of the cell; cells are NOT equal area.',
    'doy_sin': 'sin(2π·day-of-year(issue)/365.25).', 'doy_cos': 'cos(2π·day-of-year(issue)/365.25).',
    'temperature_lag2': 'ERA5 2 m air temperature, UTC-day mean of day d-2 (°C).',
    'tmax_lag2': 'ERA5 2 m temperature, hourly max on d-2 (°C).',
    'humidity_lag2': 'ERA5 2 m relative humidity, mean on d-2 (%).',
    'rhmin_lag2': 'ERA5 2 m relative humidity, hourly min on d-2 (%).',
    'wind_lag2': 'ERA5 10 m wind speed, mean on d-2 (m/s).', 'windmax_lag2': '10 m wind, hourly max on d-2 (m/s).',
    'vpd_lag2': 'Vapour-pressure deficit from hourly T and RH (Tetens), mean on d-2 (kPa).',
    'rain_3d_lag2': 'ERA5 precipitation total over d-4..d-2 (mm).',
    'rain_7d_lag2': 'Precipitation total over d-8..d-2 (mm).', 'rain_30d_lag2': 'Precipitation total over d-31..d-2 (mm).',
    'dry_days_lag2': 'Consecutive days with <1 mm precipitation ending d-2 (capped at 30).',
    'neighbor_fire_3d': 'Qualifying detections in the 8 neighbouring cells known at issue, last 3 issue days.',
    'neighbor_fire_7d': 'Same, last 7 issue days.', 'neighbor_frp_3d': 'Sum of FRP (MW) of those detections, last 3 issue days.',
    'ring2_fire_7d': 'Detections in the 16 cells two steps away, known at issue, last 7 issue days.',
    'own_fire_30d': 'Detections in the cell itself known 8..30 issue days before (0..7 is 0 by eligibility).',
    'region_fire_3d': 'Detections across the whole study bbox known at issue, last 3 issue days.',
    'eligible_new_activity': 'True if no qualifying detection in the cell was known at issue during the previous 7 issue days.',
    'eligibility_uncertain': 'True if the eligibility window or issue day includes a FIRMS archive day marked missing/suspect.',
    'label_status': '"ok" or the reason the label is withheld (never converted into a negative).',
    'target_day_firms_status': 'Archive acquisition status of the target day: ok / suspect / missing (national counts).',
    'target_cloud_overpass_pct': 'ERA5 total cloud cover at the two NOAA-20 overpass hours of the target day (%). QC ONLY.',
    'target_cloud_obscured_proxy': 'target_cloud_overpass_pct >= 80: negative labels are weak. QC ONLY, never a feature.',
    'detection_on_issue_day': 'Any qualifying detection on day d (0-24 h after issue; unknown at issue). QC ONLY.',
    'fire_obs_cutoff_utc': 'Latest acquisition time whose detections are used (issue - 3 h latency assumption).',
    'weather_valid_through_utc': 'End of the most recent weather day used (end of d-2).',
    'target_nextday_detection': 'PRIMARY LABEL: 1 if ≥1 qualifying NOAA-20 VIIRS detection in the cell on d+1, else 0; NaN if withheld.',
    'target_nextday_count': 'Secondary continuous outcome: number of qualifying detections on d+1.',
    'target_nextday_frp': 'Sum of FRP (MW) of qualifying detections on d+1.',
    'in_primary_analysis': 'Row used for modelling: land cell, eligible, eligibility certain, label ok.',
    'firms_product': 'Fire data product/vintage.', 'weather_product': 'Weather data product/vintage.',
    'dataset_version': 'FireRisk Open dataset version.', 'raw_manifest_sha256': 'SHA-256 of manifest.json listing all raw files.',
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_release(ingest_dir: Path, name: str = 'firerisk_open_norcal', log=print) -> dict:
    root = Path(ingest_dir)
    meta = json.loads((root / 'ingest_metadata.json').read_text())
    if meta.get('mode') != 'archive' or not meta.get('download_complete'):
        raise ValueError('build_release needs a complete archive ingest (fireaq ingest-archive)')
    interim, out = root / 'interim', root / 'dataset'
    out.mkdir(parents=True, exist_ok=True)
    grid = Grid(*meta['bbox'], meta['step'])
    cells = pd.read_csv(interim / 'cells.csv')
    firms = pd.read_parquet(interim / 'firms_regional.parquet')
    status = pd.read_csv(interim / 'firms_day_status.csv')
    wx = pd.read_parquet(interim / 'weather_daily.parquet')

    data = make_dataset(firms, wx, grid, meta['issue_start'], meta['issue_end'], pad_cells=NEIGHBOR_PAD_CELLS,
                        day_status=status, cells_meta=cells, keep_ineligible=True)
    land = data['cell_id'].map(cells.set_index('cell_id')['included_land_cell']).astype(bool)
    data['in_primary_analysis'] = (land & data['eligible_new_activity'] & ~data['eligibility_uncertain']
                                   & (data['label_status'] == 'ok'))
    data['firms_product'] = f"NASA FIRMS NOAA-20 VIIRS 375m yearly archive ({meta['satellite']}), version 2 (SP)"
    data['weather_product'] = 'ECMWF ERA5 via Open-Meteo archive API (models=era5)'
    data['dataset_version'] = DATASET_VERSION
    data['raw_manifest_sha256'] = sha256(root / 'manifest.json')
    data = data.sort_values(['issue_date', 'cell_id']).reset_index(drop=True)
    pq = out / f'{name}_v{DATASET_VERSION}.parquet'
    data.to_parquet(pq, index=False)
    prim = data[data['in_primary_analysis']]
    prim.head(500).to_csv(out / f'{name}_preview_500rows.csv', index=False)
    pd.DataFrame([{'column': c, 'dtype': str(data[c].dtype), 'description': DICTIONARY.get(c, '')}
                  for c in data.columns]).to_csv(out / 'data_dictionary.csv', index=False)

    # static-source mask for NRT scoring, built from TRAINING years only
    train_years = firms[pd.to_datetime(firms['acq_date']).dt.year <= 2022]
    mask = sorted(build_static_source_mask(train_years))
    (interim / 'static_source_mask_2019_2022.json').write_text(json.dumps(mask) + '\n')

    yrs = pd.to_datetime(data['target_date']).dt.year
    excl = {
        'total_cell_days': int(len(data)),
        'not_eligible_recent_detection': int((~data['eligible_new_activity']).sum()),
        'eligibility_uncertain_firms_day': int((data['eligible_new_activity'] & data['eligibility_uncertain']).sum()),
        'label_withheld_firms_target_day': int((data['eligible_new_activity'] & ~data['eligibility_uncertain']
                                               & (data['label_status'] != 'ok')).sum()),
        'non_land_cells': int((~land).sum()),
        'in_primary_analysis': int(data['in_primary_analysis'].sum()),
    }
    by_year = prim.groupby(pd.to_datetime(prim['target_date']).dt.year).agg(
        rows=('target_nextday_detection', 'size'), positives=('target_nextday_detection', 'sum'),
        prevalence=('target_nextday_detection', 'mean'),
        cloud_obscured_share=('target_cloud_obscured_proxy', 'mean'))
    pos = prim[prim['target_nextday_detection'] == 1]
    by_year['issue_day_detection_share_of_positives'] = pos.groupby(
        pd.to_datetime(pos['target_date']).dt.year)['detection_on_issue_day'].mean()
    report = {
        'dataset_file': pq.name, 'sha256': sha256(pq), 'rows': int(len(data)), 'columns': int(data.shape[1]),
        'issue_date_range': [data['issue_date'].min(), data['issue_date'].max()],
        'cells': int(data['cell_id'].nunique()), 'row_accounting': excl,
        'primary_by_target_year': by_year.reset_index().rename(columns={'target_date': 'year'}).to_dict(orient='records'),
        'primary_positive_share_with_issue_day_detection': float(pos['detection_on_issue_day'].mean()) if len(pos) else None,
        'feature_missingness_all_rows': {c: int(data[c].isna().sum()) for c in FEATURES},
        'rows_by_target_year_all': yrs.value_counts().sort_index().to_dict(),
        'land_fraction_min': float(cells['land_fraction'].min()),
        'cell_area_km2_range': [float(cells['cell_area_km2'].min()), float(cells['cell_area_km2'].max())],
        'static_source_mask_boxes': len(mask),
    }
    (out / 'coverage_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    (out / 'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in sorted(out.iterdir())
                                            if p.is_file() and p.name != 'SHA256SUMS'))
    log(f"dataset: {len(data):,} cell-days, {excl['in_primary_analysis']:,} in primary analysis, "
        f"{int(prim['target_nextday_detection'].sum()):,} positives -> {pq}")
    return report
