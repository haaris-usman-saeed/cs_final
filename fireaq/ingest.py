"""Keyless historical ingest: FIRMS yearly archive + Open-Meteo ERA5 + Natural Earth land mask.

Layout written under ``out``::

    raw/       immutable source bytes + *.meta.json sidecars (SHA-256, URL, retrieval time)
    interim/   QC'd tables: cells.csv, firms_regional.parquet, firms_day_status.csv,
               weather_daily.parquet, ingest_report.json
    manifest.json, ingest_metadata.json

Re-running is idempotent: cached raw files are verified by hash and reused.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .core import Grid
from .geo import cell_area_km2, global_cell_id, land_fraction
from .sources import (fetch_firms_country_year, fetch_natural_earth, fetch_weather_historical,
                      firms_day_status, read_firms_archive, write_manifest)

FEATURE_PAD_DAYS = 45      # longest look-back window (30-day rain/fire history + lag) with margin
NEIGHBOR_PAD_CELLS = 2     # FIRMS fetched beyond the bbox so edge cells see full neighbourhoods
MIN_LAND_FRACTION = 0.5


def build_cells(grid: Grid, rawdir: Path, manifest: list) -> pd.DataFrame:
    cells = grid.cells()
    land = fetch_natural_earth('ne_10m_land', rawdir, manifest)
    lakes = fetch_natural_earth('ne_10m_lakes', rawdir, manifest)
    cells['land_fraction'] = land_fraction(cells, grid.step, land, lakes).round(4)
    cells['cell_area_km2'] = [round(cell_area_km2(r.latitude - grid.step / 2, r.latitude + grid.step / 2,
                                                  r.longitude - grid.step / 2, r.longitude + grid.step / 2), 2)
                              for r in cells.itertuples()]
    cells['cell_global_id'] = [global_cell_id(r.latitude, r.longitude, grid.step) for r in cells.itertuples()]
    cells['included_land_cell'] = cells['land_fraction'] >= MIN_LAND_FRACTION
    return cells


def ingest_archive(bbox, step: float, start: str, end: str, out: Path, satellite: str = 'noaa20',
                   skip_weather: bool = False, log=print) -> dict:
    out = Path(out)
    raw, interim = out / 'raw', out / 'interim'
    interim.mkdir(parents=True, exist_ok=True)
    grid = Grid(*bbox, step)
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    if e < s:
        raise ValueError('end must be >= start')
    if e + timedelta(days=1) > date.today() - timedelta(days=30):
        raise ValueError('Historical archive mode needs the target day to be >30 days old')
    manifest: list = []

    cells = build_cells(grid, raw, manifest)
    cells.to_csv(interim / 'cells.csv', index=False)
    log(f'cells: {len(cells)} total, {int(cells.included_land_cell.sum())} with land fraction >= {MIN_LAND_FRACTION}')

    fire_start, fire_end = s - timedelta(days=FEATURE_PAD_DAYS), e + timedelta(days=1)
    pad = NEIGHBOR_PAD_CELLS * step
    pbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    regional, coverage, per_year = [], [], []
    for year in range(fire_start.year, fire_end.year + 1):
        path = fetch_firms_country_year(year, raw, manifest, satellite=satellite)
        reg, cov = read_firms_archive(path, pbox)
        reg['archive_year_file'] = path.name
        regional.append(reg)
        coverage.append(cov)
        per_year.append({'year': year, 'file': path.name, 'regional_rows': len(reg),
                         'national_rows': int(cov['n_all'].sum()),
                         'first_acq_date': cov['date'].min(), 'last_acq_date': cov['date'].max(),
                         'distinct_days': int(cov['date'].nunique()),
                         'version_values': sorted(reg['version'].astype(str).unique().tolist())})
        log(f'FIRMS {year}: {len(reg):,} regional rows, {per_year[-1]["national_rows"]:,} national, '
            f'{per_year[-1]["distinct_days"]} days, versions {per_year[-1]["version_values"]}')
    firms = pd.concat(regional, ignore_index=True)
    firms.to_parquet(interim / 'firms_regional.parquet', index=False)
    status = firms_day_status(pd.concat(coverage, ignore_index=True), str(fire_start), str(fire_end))
    status.to_csv(interim / 'firms_day_status.csv', index=False)

    weather_summary = None
    if not skip_weather:
        wx_start, wx_end = s - timedelta(days=FEATURE_PAD_DAYS), e + timedelta(days=1)
        frames = []
        land_cells = cells[cells.included_land_cell]
        for i, row in enumerate(land_cells.itertuples(), 1):
            frames.append(fetch_weather_historical(row.cell_id, row.latitude, row.longitude,
                                                   wx_start, wx_end, raw, manifest, api='era5'))
            if i % 8 == 0:
                log(f'weather: {i}/{len(land_cells)} cells')
                write_manifest(out / 'manifest.json', manifest)
        wx = pd.concat(frames, ignore_index=True)
        wx.to_parquet(interim / 'weather_daily.parquet', index=False)
        weather_summary = {
            'cells': int(wx['cell_id'].nunique()), 'days': int(wx['date'].nunique()),
            'first_day': wx['date'].min(), 'last_day': wx['date'].max(),
            'incomplete_days': int((wx['n_hours'] != 24).sum()),
            'missing_by_column': {c: int(wx[c].isna().sum()) for c in wx.columns if wx[c].isna().any()},
        }

    write_manifest(out / 'manifest.json', manifest)
    report = {
        'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'bbox': list(bbox), 'step': step, 'issue_start': start, 'issue_end': end,
        'fire_window': [str(fire_start), str(fire_end)], 'padded_fire_bbox': list(pbox),
        'satellite': satellite,
        'firms_per_year': per_year,
        'firms_regional_rows_total': len(firms),
        'firms_day_status_counts': status['status'].value_counts().to_dict(),
        'firms_problem_days': status[status.status != 'ok'][['date', 'n_day', 'n_night', 'n_all', 'status']]
                              .to_dict(orient='records'),
        'weather': weather_summary,
        'raw_files': len({m['file'] for m in manifest}),
        'network_downloads_this_run': sum(1 for m in manifest if not m.get('cache_hit')),
        'cache_hits_this_run': sum(1 for m in manifest if m.get('cache_hit')),
    }
    (interim / 'ingest_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    meta = {
        'mode': 'archive', 'bbox': list(bbox), 'step': step, 'issue_start': start, 'issue_end': end,
        'fire_start': str(fire_start), 'fire_end': str(fire_end), 'satellite': satellite,
        'fire_source_products': [f'FIRMS yearly country archive {satellite} (VIIRS 375 m)'],
        'weather_source': 'Open-Meteo ERA5 (models=era5) reanalysis, hourly, UTC',
        'weather_data_warning': 'ERA5 reanalysis is NOT an as-issued operational forecast',
        'download_complete': not skip_weather, 'live': False, 'raw_manifest_file': 'manifest.json',
    }
    (out / 'ingest_metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return report
