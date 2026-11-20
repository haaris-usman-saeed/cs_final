"""v0.3 multi-region keyless ingest.

data/v03/
  raw/firms        NOAA-20 2018-2025 + S-NPP 2012-2017 yearly US archives (hash-verified, shared)
  raw/weather      NASA POWER daily JSON per cell
  raw/landcover    ESA WorldCover 2020 per-cell class counts
  raw/terrain      Copernicus DEM GLO-90 per-cell stats (AWS COG)
  raw/geo          Natural Earth land/lakes
  interim/         firms_day_status.csv, <region>/{cells.csv, firms_regional.parquet, weather.parquet}
  manifest.json, ingest_report.json
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .core import Grid, clean_firms
from .geo import cell_area_km2, global_cell_id, land_fraction
from .regions import REGIONS, STEP
from .sources import (cached_entry, dem_terrain, fetch_firms_country_year, fetch_natural_earth,
                      fetch_power_daily, firms_day_status, worldcover_fractions, write_manifest)

N20_YEARS = range(2018, 2026)
SNPP_CLIM_YEARS = range(2012, 2018)
WEATHER_START, WEATHER_END = date(2018, 9, 1), date(2025, 12, 31)
PAD_CELLS = 2
MIN_LAND = 0.5


def _link_previous(raw: Path, prev: Path) -> None:
    """Reuse already-downloaded, hash-sidecar'd raw files (hard links, no re-download)."""
    for sub in ('firms', 'geo'):
        src = prev / sub
        if not src.exists():
            continue
        (raw / sub).mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            dst = raw / sub / f.name
            if not dst.exists() and (f.suffix in ('.csv', '.geojson') or f.name.endswith('.meta.json')):
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy2(f, dst)


def region_cells(region: str, rawdir: Path, manifest: list) -> pd.DataFrame:
    grid = Grid(*REGIONS[region]['bbox'], STEP)
    c = grid.cells()
    land = fetch_natural_earth('ne_10m_land', rawdir, manifest)
    lakes = fetch_natural_earth('ne_10m_lakes', rawdir, manifest)
    c['land_fraction'] = land_fraction(c, STEP, land, lakes).round(4)
    c['cell_area_km2'] = [round(cell_area_km2(r.latitude - STEP/2, r.latitude + STEP/2,
                                              r.longitude - STEP/2, r.longitude + STEP/2), 2) for r in c.itertuples()]
    c['cell_global_id'] = [global_cell_id(r.latitude, r.longitude, STEP) for r in c.itertuples()]
    c['region'] = region
    c['cell_key'] = region + ':' + c['cell_id']
    c['included_land_cell'] = c['land_fraction'] >= MIN_LAND
    return c


def snpp_climatology(snpp_frames: dict, region: str, cells: pd.DataFrame) -> pd.DataFrame:
    """Fire-days per cell (annual mean and per calendar month) from S-NPP 2012-2017, type 0, conf n/h."""
    grid = Grid(*REGIONS[region]['bbox'], STEP)
    days = []
    for yr, df in snpp_frames.items():
        c = clean_firms(df, grid)
        days.append(c[['cell_id', 'observed_date']].drop_duplicates())
    fd = pd.concat(days, ignore_index=True)
    n_years = len(snpp_frames)
    fd['month'] = fd['observed_date'].dt.month
    annual = fd.groupby('cell_id').size() / n_years
    monthly = fd.groupby(['cell_id', 'month']).size().unstack(fill_value=0) / n_years
    monthly = monthly.reindex(columns=range(1, 13), fill_value=0.0)
    monthly.columns = [f'clim_fire_days_m{m:02d}' for m in monthly.columns]
    out = cells[['cell_id']].merge(annual.rename('clim_fire_days_year').reset_index(), on='cell_id', how='left')
    out = out.merge(monthly.reset_index(), on='cell_id', how='left').fillna(0.0)
    return out


def ingest_v3(out: Path = Path('data/v03'), prev: Path | None = Path('data/norcal/raw'),
              regions: list[str] | None = None, log=print) -> dict:
    out = Path(out)
    raw, interim = out / 'raw', out / 'interim'
    interim.mkdir(parents=True, exist_ok=True)
    regions = regions or list(REGIONS)
    if prev is not None and Path(prev).exists():
        _link_previous(raw, Path(prev))
    manifest: list = []
    boxes = {}
    for r in regions:
        w, s, e, n = REGIONS[r]['bbox']
        p = PAD_CELLS * STEP
        boxes[r] = (w - p, s - p, e + p, n + p)

    # --- NOAA-20 detections (padded per-region extracts) + national day status
    regional = {r: [] for r in regions}
    coverage, per_year = [], []
    for y in N20_YEARS:
        path = fetch_firms_country_year(y, raw, manifest, satellite='noaa20')
        df = pd.read_csv(path, dtype={'acq_time': str, 'version': str, 'confidence': str})
        cov = df.groupby(['acq_date', 'daynight']).size().unstack(fill_value=0).reindex(columns=['D', 'N'], fill_value=0)
        cov = cov.rename(columns={'D': 'n_day', 'N': 'n_night'})
        cov['n_all'] = cov.n_day + cov.n_night
        coverage.append(cov.reset_index().rename(columns={'acq_date': 'date'}))
        for r in regions:
            w, s, e, n = boxes[r]
            regional[r].append(df[df.latitude.between(s, n) & df.longitude.between(w, e)])
        per_year.append({'satellite': 'noaa20', 'year': y, 'file': path.name, 'national_rows': len(df),
                         'distinct_days': int(df.acq_date.nunique()),
                         'versions': sorted(df['version'].astype(str).unique().tolist())})
        log(f'NOAA-20 {y}: {len(df):,} national rows')
    status = firms_day_status(pd.concat(coverage, ignore_index=True), str(WEATHER_START), str(WEATHER_END))
    status.to_csv(interim / 'firms_day_status.csv', index=False)

    # --- S-NPP 2012-2017 for the pre-study climatology only
    snpp = {r: {} for r in regions}
    for y in SNPP_CLIM_YEARS:
        path = fetch_firms_country_year(y, raw, manifest, satellite='snpp')
        df = pd.read_csv(path, dtype={'acq_time': str, 'version': str, 'confidence': str})
        for r in regions:
            w, s, e, n = REGIONS[r]['bbox']
            snpp[r][y] = df[df.latitude.between(s, n) & df.longitude.between(w, e)]
        per_year.append({'satellite': 'snpp', 'year': y, 'file': path.name, 'national_rows': len(df),
                         'distinct_days': int(df.acq_date.nunique()),
                         'versions': sorted(df['version'].astype(str).unique().tolist())})
        log(f'S-NPP {y}: {len(df):,} national rows (climatology only)')

    region_reports = {}
    for r in regions:
        rd = interim / r
        rd.mkdir(parents=True, exist_ok=True)
        cells = region_cells(r, raw, manifest)
        land = cells[cells.included_land_cell].copy()
        lc = worldcover_fractions(land, STEP, raw, manifest)
        dem = dem_terrain(land, STEP, raw, manifest)
        clim = snpp_climatology(snpp[r], r, cells)
        cells = cells.merge(lc, on='cell_key', how='left').merge(dem, on='cell_key', how='left') \
                     .merge(clim, on='cell_id', how='left')
        cells.to_csv(rd / 'cells.csv', index=False)
        pd.concat(regional[r], ignore_index=True).to_parquet(rd / 'firms_regional.parquet', index=False)
        frames = []
        for i, c in enumerate(land.itertuples(), 1):
            frames.append(fetch_power_daily(c.cell_id, c.latitude, c.longitude, WEATHER_START, WEATHER_END,
                                            raw / r, manifest))
            if i % 16 == 0:
                log(f'{r}: POWER {i}/{len(land)} cells')
        wx = pd.concat(frames, ignore_index=True)
        wx.to_parquet(rd / 'weather.parquet', index=False)
        region_reports[r] = {
            'role': REGIONS[r]['role'], 'bbox': REGIONS[r]['bbox'], 'cells': len(cells),
            'land_cells': int(cells.included_land_cell.sum()),
            'land_fraction_min_included': float(land.land_fraction.min()),
            'firms_padded_rows_2018_2025': int(sum(len(x) for x in regional[r])),
            'weather_days': int(wx['date'].nunique()),
            'weather_missing_by_column': {k: int(v) for k, v in wx.isna().sum().items() if v},
            'power_sources_seen': sorted(wx['power_sources'].unique().tolist()),
            'snpp_clim_fire_days_year_mean': float(clim.clim_fire_days_year.mean()),
        }
        log(f'{r}: done {region_reports[r]}')
        write_manifest(out / 'manifest.json', manifest)

    write_manifest(out / 'manifest.json', manifest)
    report = {'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'step': STEP,
              'weather_window': [str(WEATHER_START), str(WEATHER_END)], 'archives': per_year,
              'firms_day_status_counts': status['status'].value_counts().to_dict(),
              'regions': region_reports, 'raw_files': len({m['file'] for m in manifest}),
              'network_downloads_this_run': sum(1 for m in manifest if not m.get('cache_hit')),
              'cache_hits_this_run': sum(1 for m in manifest if m.get('cache_hit'))}
    (out / 'ingest_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    return report
