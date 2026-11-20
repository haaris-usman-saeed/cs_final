"""v0.3 cell × issue-day builder (NASA POWER weather, static layers, live-computable fire history).

Label, eligibility and timing rules are identical to v0.2 (``fireaq.core``):
issue 00:00 UTC on d; target = qualifying NOAA-20 detection on UTC day d+1; eligible if no detection known
at issue during the previous 7 issue days; fire features use acq_time + 3 h <= issue; weather uses day d-2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import (CLOUD_OBSCURED_PCT, ELIGIBILITY_DAYS, FIRE_LATENCY_H, Grid, _box, _rolling, clean_firms)

DATASET_VERSION = '0.3.0'
WEATHER_LAG_DAYS = 2
PAD_DAYS = 100        # 90-day rain window + lag + margin

STATIC_LC = ['lc_tree', 'lc_shrub', 'lc_grass', 'lc_crop', 'lc_built', 'lc_bare', 'lc_wetland']
SET_A = (['doy_sin', 'doy_cos',
          't2m_lag2', 't2m_max_lag2', 'rh2m_lag2', 'vpd_lag2', 'ws10m_lag2', 'ws10m_max_lag2',
          'rain_3d_lag2', 'rain_7d_lag2', 'rain_30d_lag2', 'rain_90d_lag2', 'dry_days_lag2',
          'gwettop_lag2', 'gwetroot_lag2', 'gwetroot_chg30_lag2']
         + STATIC_LC + ['elev_mean_m', 'elev_std_m'])
FIRE_B = ['neighbor_fire_3d', 'neighbor_fire_7d', 'neighbor_frp_3d', 'ring2_fire_7d', 'region_mean_fire_3d',
          'clim_fire_days_year', 'clim_fire_days_month']
SET_B = SET_A + FIRE_B
FEATURE_SETS = {'A_no_fire_info': SET_A, 'B_plus_fire_history': SET_B}
LOG_FEATURES = FIRE_B + ['rain_3d_lag2', 'rain_7d_lag2', 'rain_30d_lag2', 'rain_90d_lag2', 'elev_std_m']
TARGET = 'target_nextday_detection'


def _dry_run(w: np.ndarray) -> float:
    for i in range(len(w) - 1, -1, -1):
        if np.isnan(w[i]):
            return np.nan
        if w[i] >= 1.0:
            return float(len(w) - 1 - i)
    return float(len(w))


def make_dataset_v3(fires_raw: pd.DataFrame, weather: pd.DataFrame, cells: pd.DataFrame, region: str,
                    bbox, step: float, start: str, end: str, *, pad_cells: int = 2,
                    day_status: pd.DataFrame | None = None, static_mask: set | None = None,
                    predict_only: bool = False, keep_ineligible: bool = False) -> pd.DataFrame:
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError('start after end')
    if fires_raw is None or fires_raw.empty:
        raise ValueError('FIRMS data is empty: refusing to convert missing acquisitions into zero observations')
    grid = Grid(*bbox, step)
    pgrid = grid.padded(pad_cells)
    ox, oy = round((grid.west - pgrid.west) / step), round((grid.south - pgrid.south) / step)
    fires = clean_firms(fires_raw, pgrid, static_mask=static_mask)
    dates = pd.date_range(start_ts - pd.Timedelta(days=PAD_DAYS), end_ts + pd.Timedelta(days=1))
    T, NY, NX = len(dates), pgrid.ny, pgrid.nx
    didx = {d: i for i, d in enumerate(dates)}

    def acc(col, w=None):
        a = np.zeros((T, NY, NX))
        t = fires[col].map(didx)
        ok = t.notna().to_numpy()
        ww = np.ones(ok.sum()) if w is None else fires.loc[ok, w].to_numpy(float)
        np.add.at(a, (t[ok].astype(int).to_numpy(), fires.loc[ok, 'iy'].to_numpy(), fires.loc[ok, 'ix'].to_numpy()), ww)
        return a

    known, known_frp = acc('first_usable_issue'), acc('first_usable_issue', 'frp')
    obs, obs_frp = acc('observed_date'), acc('observed_date', 'frp')
    k3, k7, f3 = _rolling(known, 3), _rolling(known, ELIGIBILITY_DAYS), _rolling(known_frp, 3)
    feats = {'own_known_7d': k7, 'neighbor_fire_3d': _box(k3, 1) - k3, 'neighbor_fire_7d': _box(k7, 1) - k7,
             'neighbor_frp_3d': _box(f3, 1) - f3, 'ring2_fire_7d': _box(k7, 2) - _box(k7, 1),
             'obs_count': obs, 'obs_frp': obs_frp}
    core = k3[:, oy:oy+grid.ny, ox:ox+grid.nx]
    land_ids = set(cells.loc[cells.included_land_cell, 'cell_id'])
    gc = grid.cells()
    land_mask = gc['cell_id'].isin(land_ids).to_numpy().reshape(grid.ny, grid.nx)
    region_mean = np.where(np.isnan(core).all(axis=(1, 2)), np.nan,
                           np.nansum(core * land_mask, axis=(1, 2)) / max(1, land_mask.sum()))
    rows = []
    for c in gc[gc.cell_id.isin(land_ids)].itertuples():
        y, x = c.iy + oy, c.ix + ox
        d = {k: v[:, y, x] for k, v in feats.items()}
        d.update(cell_id=c.cell_id, date=dates, region_mean_fire_3d=region_mean)
        rows.append(pd.DataFrame(d))
    panel = pd.concat(rows, ignore_index=True)
    g = panel.groupby('cell_id', sort=False)
    panel['target_nextday_count'] = g['obs_count'].shift(-1)
    panel['target_nextday_frp'] = g['obs_frp'].shift(-1)
    panel['detection_on_issue_day'] = panel['obs_count'] > 0

    # ---- weather (POWER daily UTC), complete values only, lagged
    need = {'cell_id', 'date', 't2m', 't2m_max', 'rh2m', 'precip_mm', 'ws10m', 'ws10m_max', 'gwettop', 'gwetroot',
            'vpd_kpa', 'cloud_pct'}
    if not need.issubset(weather.columns):
        raise ValueError(f'weather missing columns: {need - set(weather.columns)}')
    wx = weather.copy()
    wx['date'] = pd.to_datetime(wx['date']).dt.normalize()
    if wx.duplicated(['cell_id', 'date']).any():
        raise ValueError('weather has duplicated cell/day rows; resolve upstream')
    wx = panel[['cell_id', 'date']].merge(wx, on=['cell_id', 'date'], how='left')
    wg = wx.groupby('cell_id', sort=False)
    L = WEATHER_LAG_DAYS
    for src, dst in [('t2m', 't2m_lag2'), ('t2m_max', 't2m_max_lag2'), ('rh2m', 'rh2m_lag2'), ('vpd_kpa', 'vpd_lag2'),
                     ('ws10m', 'ws10m_lag2'), ('ws10m_max', 'ws10m_max_lag2'), ('gwettop', 'gwettop_lag2'),
                     ('gwetroot', 'gwetroot_lag2')]:
        panel[dst] = wg[src].shift(L).to_numpy()
    panel['gwetroot_chg30_lag2'] = (wg['gwetroot'].shift(L) - wg['gwetroot'].shift(L + 30)).to_numpy()
    for days in (3, 7, 30, 90):
        s = wg['precip_mm'].transform(lambda v: v.rolling(days, min_periods=days).sum())
        panel[f'rain_{days}d_lag2'] = s.groupby(wx['cell_id']).shift(L).to_numpy()
    dry = wg['precip_mm'].transform(lambda v: v.rolling(30, min_periods=30).apply(_dry_run, raw=True))
    panel['dry_days_lag2'] = dry.groupby(wx['cell_id']).shift(L).to_numpy()
    panel['target_cloud_pct'] = wg['cloud_pct'].shift(-1).to_numpy()
    panel['weather_source_lag2'] = wg['power_sources'].shift(L).to_numpy() if 'power_sources' in wx else None

    # ---- issue window, calendar, statics
    panel = panel[panel['date'].between(start_ts, end_ts)].copy()
    doy = panel['date'].dt.dayofyear
    panel['doy_sin'], panel['doy_cos'] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    month = panel['date'].dt.month
    st = cells.set_index('cell_id')
    static_cols = ['cell_key', 'cell_global_id', 'region', 'latitude', 'longitude', 'land_fraction',
                   'cell_area_km2', 'elev_mean_m', 'elev_std_m', 'clim_fire_days_year'] + STATIC_LC
    panel = panel.join(st[static_cols], on='cell_id')
    mcols = [f'clim_fire_days_m{m:02d}' for m in range(1, 13)]
    mt = st[mcols].to_numpy()
    pos = st.index.get_indexer(panel['cell_id'])
    panel['clim_fire_days_month'] = mt[pos, month.to_numpy() - 1]
    panel['issue_date'] = panel['date'].dt.strftime('%Y-%m-%d')
    panel['target_date'] = (panel['date'] + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    panel['issue_time_utc'] = panel['date'].dt.strftime('%Y-%m-%dT00:00:00Z')
    panel['target_start_utc'] = (panel['date'] + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%dT00:00:00Z')
    panel['target_end_utc'] = (panel['date'] + pd.Timedelta(days=2)).dt.strftime('%Y-%m-%dT00:00:00Z')
    panel['fire_obs_cutoff_utc'] = (panel['date'] - pd.Timedelta(hours=FIRE_LATENCY_H)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    panel['weather_valid_through_utc'] = (panel['date'] - pd.Timedelta(days=L-1)).dt.strftime('%Y-%m-%dT00:00:00Z')

    # ---- eligibility / QC
    panel['eligible_new_activity'] = panel['own_known_7d'].eq(0)
    panel['label_status'] = 'ok'
    panel['eligibility_uncertain'] = False
    if day_status is not None:
        s = day_status.assign(date=pd.to_datetime(day_status['date'])).set_index('date')['status']
        panel['target_day_firms_status'] = (panel['date'] + pd.Timedelta(days=1)).map(s).fillna('missing').to_numpy()
        bad = set(s[s != 'ok'].index)
        win = pd.Series(False, index=panel.index)
        for k in range(ELIGIBILITY_DAYS + 1):
            win |= (panel['date'] - pd.Timedelta(days=k)).isin(bad)
        panel['eligibility_uncertain'] = win.to_numpy()
        panel.loc[panel['target_day_firms_status'] != 'ok', 'label_status'] = 'firms_target_day_not_ok'
    else:
        panel['target_day_firms_status'] = 'not_checked'
    panel['target_cloud_obscured_proxy'] = panel['target_cloud_pct'] >= CLOUD_OBSCURED_PCT
    panel = panel.dropna(subset=SET_B)
    if predict_only:
        for c in [TARGET, 'target_nextday_count', 'target_nextday_frp']:
            panel[c] = np.nan
        panel['label_status'] = 'future'
    else:
        panel = panel[panel['target_nextday_count'].notna()].copy()
        panel[TARGET] = (panel['target_nextday_count'] > 0).astype(float)
        bad = panel['label_status'] != 'ok'
        panel.loc[bad, [TARGET, 'target_nextday_count', 'target_nextday_frp']] = np.nan
    if not keep_ineligible:
        panel = panel[panel['eligible_new_activity'] & ~panel['eligibility_uncertain']].copy()
    ids = ['region', 'cell_key', 'cell_id', 'cell_global_id', 'issue_date', 'target_date', 'issue_time_utc',
           'target_start_utc', 'target_end_utc', 'latitude', 'longitude', 'land_fraction', 'cell_area_km2']
    qc = ['eligible_new_activity', 'eligibility_uncertain', 'label_status', 'target_day_firms_status',
          'target_cloud_pct', 'target_cloud_obscured_proxy', 'detection_on_issue_day', 'fire_obs_cutoff_utc',
          'weather_valid_through_utc', 'weather_source_lag2']
    out = panel[ids + SET_B + qc + [TARGET, 'target_nextday_count', 'target_nextday_frp']]
    return out.sort_values(['issue_date', 'cell_key']).reset_index(drop=True)
