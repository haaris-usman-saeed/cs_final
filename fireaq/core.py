"""Create an *observed-detection* research dataset with strictly past-only features.

This project DOES NOT infer ignition, fire spread, or true wildfire absence.
Target: a qualifying satellite detection in the grid cell during UTC day d+1 (24-48 h after the
00:00 UTC issue time on day d), among cells with no detection *known at issue time* during the
preceding seven issue days.

Timing conventions (all UTC):
* issue time            = d 00:00
* fire features         = detections with acquisition time + FIRE_LATENCY_H <= issue time
* weather features      = complete UTC days up to and including d - WEATHER_LAG_DAYS
* target window         = [d+1 00:00, d+2 00:00)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FIRE_LATENCY_H = 3      # assumed NRT availability latency; SP products are retrospective
WEATHER_LAG_DAYS = 2    # conservative: the most recent weather day used is d-2
ELIGIBILITY_DAYS = 7
HISTORY_DAYS = 30
PAD_DAYS = 45
CLOUD_OBSCURED_PCT = 80.0
DATASET_VERSION = '0.2.0'

WEATHER_FEATURES = ['latitude', 'longitude', 'doy_sin', 'doy_cos',
                    'temperature_lag2', 'tmax_lag2', 'humidity_lag2', 'rhmin_lag2',
                    'wind_lag2', 'windmax_lag2', 'vpd_lag2',
                    'rain_3d_lag2', 'rain_7d_lag2', 'rain_30d_lag2', 'dry_days_lag2']
FIRE_FEATURES = ['neighbor_fire_3d', 'neighbor_fire_7d', 'neighbor_frp_3d', 'ring2_fire_7d',
                 'own_fire_30d', 'region_fire_3d']
FEATURES = WEATHER_FEATURES + FIRE_FEATURES
COUNT_FEATURES = FIRE_FEATURES + ['rain_3d_lag2', 'rain_7d_lag2', 'rain_30d_lag2']


@dataclass(frozen=True)
class Grid:
    west: float
    south: float
    east: float
    north: float
    step: float = 0.5

    def __post_init__(self):
        if not -180 <= self.west < self.east <= 180 or not -90 <= self.south < self.north <= 90:
            raise ValueError('Invalid bounding box')
        if self.step <= 0 or self.step > 2:
            raise ValueError('step must be >0 and <=2 degrees')
        if abs((self.east-self.west)/self.step - round((self.east-self.west)/self.step)) > 1e-7:
            raise ValueError('bbox width must be an integer multiple of grid step')
        if abs((self.north-self.south)/self.step - round((self.north-self.south)/self.step)) > 1e-7:
            raise ValueError('bbox height must be an integer multiple of grid step')

    @property
    def nx(self): return round((self.east-self.west)/self.step)

    @property
    def ny(self): return round((self.north-self.south)/self.step)

    def padded(self, n: int) -> 'Grid':
        p = n * self.step
        return Grid(max(-180, self.west-p), max(-90, self.south-p),
                    min(180, self.east+p), min(90, self.north+p), self.step)

    def cells(self):
        return pd.DataFrame([{
            'cell_id': f'r{y}_c{x}', 'ix': x, 'iy': y,
            'latitude': self.south + (y+0.5)*self.step,
            'longitude': self.west + (x+0.5)*self.step,
        } for y in range(self.ny) for x in range(self.nx)])

    def index(self, lat, lon):
        x = np.floor((pd.to_numeric(lon, errors='coerce').to_numpy(dtype=float)-self.west)/self.step)
        y = np.floor((pd.to_numeric(lat, errors='coerce').to_numpy(dtype=float)-self.south)/self.step)
        good = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < self.nx) & (y >= 0) & (y < self.ny)
        return x, y, good

    def assign(self, lat: pd.Series, lon: pd.Series):
        x, y, good = self.index(lat, lon)
        names = np.full(len(x), None, dtype=object)
        names[good] = [f'r{int(yy)}_c{int(xx)}' for xx, yy in zip(x[good], y[good])]
        return names


def _static_key(lat, lon):
    return (np.round(pd.to_numeric(lat).to_numpy(float), 2).astype(str) + '_'
            + np.round(pd.to_numeric(lon).to_numpy(float), 2).astype(str))


def build_static_source_mask(archive: pd.DataFrame) -> set[str]:
    """0.01° boxes (plus their 8 neighbours) holding archive detections typed as static land
    sources (type 2) or offshore (type 3). Used only for NRT data, which lacks ``type``."""
    if 'type' not in archive:
        raise ValueError('archive has no type column')
    st = archive[pd.to_numeric(archive['type'], errors='coerce').isin([2, 3])]
    lat = np.round(pd.to_numeric(st['latitude']).to_numpy(float), 2)
    lon = np.round(pd.to_numeric(st['longitude']).to_numpy(float), 2)
    keys = set()
    for dy in (-0.01, 0, 0.01):
        for dx in (-0.01, 0, 0.01):
            keys.update(np.char.add(np.char.add(np.round(lat+dy, 2).astype(str), '_'),
                                    np.round(lon+dx, 2).astype(str)))
    return keys


def clean_firms(df: pd.DataFrame, grid: Grid, static_mask: set[str] | None = None) -> pd.DataFrame:
    """Parse VIIRS FIRMS CSV, retaining nominal/high-confidence presumed vegetation detections.

    * ``type`` present (archive/SP): keep ``type == 0``.
    * ``type`` absent (NRT): drop detections inside ``static_mask`` if one is supplied.
    """
    base = ['latitude', 'longitude', 'acq_date', 'acq_time', 'confidence', 'frp']
    absent = set(base) - set(df.columns)
    if absent:
        raise ValueError(f'FIRMS missing required columns: {absent}')
    x = df.copy()
    x['latitude'] = pd.to_numeric(x['latitude'], errors='coerce')
    x['longitude'] = pd.to_numeric(x['longitude'], errors='coerce')
    x['frp'] = pd.to_numeric(x['frp'], errors='coerce')
    x = x[x['confidence'].astype(str).str.lower().isin(['n', 'h', 'nominal', 'high'])]
    if 'type' in x:
        x = x[pd.to_numeric(x['type'], errors='coerce') == 0]
    elif static_mask is not None and len(x):
        x = x[~pd.Series(_static_key(x['latitude'], x['longitude']), index=x.index).isin(static_mask)]
    t = x['acq_time'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
    x['observed_utc'] = pd.to_datetime(x['acq_date'].astype(str)+' '+t.str[:2]+':'+t.str[2:],
                                       utc=True, errors='coerce')
    x = x[x['observed_utc'].notna() & x['frp'].ge(0)]
    xi, yi, good = grid.index(x['latitude'], x['longitude'])
    x = x[good].copy()
    x['ix'], x['iy'] = xi[good].astype(int), yi[good].astype(int)
    x['cell_id'] = [f'r{a}_c{b}' for a, b in zip(x['iy'], x['ix'])]
    x = x.drop_duplicates(['observed_utc', 'latitude', 'longitude'])
    x['observed_date'] = x['observed_utc'].dt.tz_localize(None).dt.normalize()
    x['first_usable_issue'] = ((x['observed_utc'] + pd.Timedelta(hours=FIRE_LATENCY_H))
                               .dt.ceil('D').dt.tz_localize(None))
    return x[['cell_id', 'ix', 'iy', 'observed_utc', 'observed_date', 'first_usable_issue', 'frp']] \
        .reset_index(drop=True)


def _rolling(a: np.ndarray, k: int) -> np.ndarray:
    """Trailing k-day sum along axis 0; NaN until k days are available."""
    c = np.cumsum(np.nan_to_num(a), axis=0, dtype=float)
    out = np.full(a.shape, np.nan)
    out[k-1:] = c[k-1:] - np.concatenate([np.zeros((1,)+a.shape[1:]), c[:-k]], axis=0)
    return out


def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r+1)x(2r+1) spatial window (axes 1, 2), zero beyond the array edge."""
    p = np.pad(a, ((0, 0), (r, r), (r, r)))
    out = np.zeros_like(a, dtype=float)
    for dy in range(2*r+1):
        for dx in range(2*r+1):
            out += p[:, dy:dy+a.shape[1], dx:dx+a.shape[2]]
    return out


def make_dataset(fires_raw: pd.DataFrame, weather: pd.DataFrame, grid: Grid,
                 start: str, end: str, *, predict_only=False, pad_cells: int = 0,
                 day_status: pd.DataFrame | None = None, cells_meta: pd.DataFrame | None = None,
                 static_mask: set[str] | None = None, keep_ineligible: bool = False) -> pd.DataFrame:
    """Create cell × issue-day records; target uses NEXT day, predictors known at issue midnight.

    ``pad_cells``: FIRMS detections up to this many cells beyond ``grid`` feed neighbour features.
    ``day_status``: per-UTC-day FIRMS acquisition status (``ok``/``suspect``/``missing``); target days
    that are not ``ok`` get a missing label (never a negative) and an explanatory ``label_status``.
    """
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError('start after end')
    if fires_raw is None or fires_raw.empty:
        raise ValueError('FIRMS data is empty: refusing to convert missing acquisitions into zero observations')
    pgrid = grid.padded(pad_cells)
    ox = round((grid.west - pgrid.west) / grid.step)
    oy = round((grid.south - pgrid.south) / grid.step)
    fires = clean_firms(fires_raw, pgrid, static_mask=static_mask)
    dates = pd.date_range(start_ts - pd.Timedelta(days=PAD_DAYS), end_ts + pd.Timedelta(days=1))
    T, NY, NX = len(dates), pgrid.ny, pgrid.nx
    didx = {d: i for i, d in enumerate(dates)}

    def accumulate(day_col, weights=None):
        arr = np.zeros((T, NY, NX))
        t = fires[day_col].map(didx)
        ok = t.notna().to_numpy()
        w = np.ones(ok.sum()) if weights is None else fires.loc[ok, weights].to_numpy(float)
        np.add.at(arr, (t[ok].astype(int).to_numpy(), fires.loc[ok, 'iy'].to_numpy(),
                        fires.loc[ok, 'ix'].to_numpy()), w)
        return arr

    known, known_frp = accumulate('first_usable_issue'), accumulate('first_usable_issue', 'frp')
    obs, obs_frp = accumulate('observed_date'), accumulate('observed_date', 'frp')
    k3, k7, k30 = _rolling(known, 3), _rolling(known, ELIGIBILITY_DAYS), _rolling(known, HISTORY_DAYS)
    f3 = _rolling(known_frp, 3)
    feats = {
        'own_known_7d': k7,
        'neighbor_fire_3d': _box(k3, 1) - k3,
        'neighbor_fire_7d': _box(k7, 1) - k7,
        'neighbor_frp_3d': _box(f3, 1) - f3,
        'ring2_fire_7d': _box(k7, 2) - _box(k7, 1),
        'own_fire_30d': k30 - k7,
        'obs_count': obs, 'obs_frp': obs_frp,
    }
    core = (slice(None), slice(oy, oy+grid.ny), slice(ox, ox+grid.nx))
    region3 = np.nansum(k3[core], axis=(1, 2))
    region3[np.isnan(k3[core]).all(axis=(1, 2))] = np.nan
    cells = grid.cells()
    rows = []
    for c in cells.itertuples():
        y, x = c.iy + oy, c.ix + ox
        d = {k: v[:, y, x] for k, v in feats.items()}
        d.update(cell_id=c.cell_id, date=dates, region_fire_3d=region3)
        rows.append(pd.DataFrame(d))
    panel = pd.concat(rows, ignore_index=True)
    g = panel.groupby('cell_id', sort=False)
    panel['target_nextday_count'] = g['obs_count'].shift(-1)
    panel['target_nextday_frp'] = g['obs_frp'].shift(-1)
    panel['detection_on_issue_day'] = panel['obs_count'] > 0   # post-issue QC only, not a feature

    # ---- weather: strictly complete UTC days, lagged
    must = {'cell_id', 'date', 'temp_c', 'rh_pct', 'precip_mm', 'wind_ms', 'n_hours'}
    if not must.issubset(weather.columns):
        raise ValueError(f'weather missing columns: {must - set(weather.columns)}')
    wx = weather.copy()
    wx['date'] = pd.to_datetime(wx['date']).dt.normalize()
    dup = wx.duplicated(['cell_id', 'date'], keep=False)
    if dup.any():
        raise ValueError(f'weather has {int(dup.sum())} duplicated cell/day rows; resolve upstream')
    for col in ['tmax_c', 'rhmin_pct', 'windmax_ms', 'vpd_kpa', 'cloud_pct', 'cloud_overpass_pct']:
        if col not in wx:
            wx[col] = np.nan
    wx = panel[['cell_id', 'date']].merge(wx, on=['cell_id', 'date'], how='left')
    wvars = ['temp_c', 'tmax_c', 'rh_pct', 'rhmin_pct', 'precip_mm', 'wind_ms', 'windmax_ms', 'vpd_kpa']
    wx.loc[wx['n_hours'] != 24, wvars] = np.nan
    wg = wx.groupby('cell_id', sort=False)
    lag = WEATHER_LAG_DAYS
    for src, dst in [('temp_c', 'temperature_lag2'), ('tmax_c', 'tmax_lag2'), ('rh_pct', 'humidity_lag2'),
                     ('rhmin_pct', 'rhmin_lag2'), ('wind_ms', 'wind_lag2'), ('windmax_ms', 'windmax_lag2'),
                     ('vpd_kpa', 'vpd_lag2')]:
        panel[dst] = wg[src].shift(lag).to_numpy()
    for days in (3, 7, 30):
        s = wg['precip_mm'].transform(lambda v: v.rolling(days, min_periods=days).sum())
        panel[f'rain_{days}d_lag2'] = s.groupby(wx['cell_id']).shift(lag).to_numpy()

    def dry_run(w: np.ndarray) -> float:
        """Consecutive days with <1 mm ending at the window's last day (capped at the window)."""
        for i in range(len(w) - 1, -1, -1):
            if np.isnan(w[i]):
                return np.nan
            if w[i] >= 1.0:
                return float(len(w) - 1 - i)
        return float(len(w))
    dry = wg['precip_mm'].transform(lambda v: v.rolling(HISTORY_DAYS, min_periods=HISTORY_DAYS)
                                    .apply(dry_run, raw=True))
    panel['dry_days_lag2'] = dry.groupby(wx['cell_id']).shift(lag).to_numpy()
    panel['target_cloud_overpass_pct'] = wg['cloud_overpass_pct'].shift(-1).to_numpy()

    # ---- restrict to issue window
    panel = panel[panel['date'].between(start_ts, end_ts)].copy()
    panel = panel.merge(cells[['cell_id', 'latitude', 'longitude']], on='cell_id', how='left')
    doy = panel['date'].dt.dayofyear
    panel['doy_sin'] = np.sin(2*np.pi*doy/365.25)
    panel['doy_cos'] = np.cos(2*np.pi*doy/365.25)
    panel['issue_date'] = panel['date'].dt.strftime('%Y-%m-%d')
    panel['target_date'] = (panel['date'] + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    panel['issue_time_utc'] = panel['date'].dt.strftime('%Y-%m-%dT00:00:00Z')
    panel['target_start_utc'] = (panel['date'] + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%dT00:00:00Z')
    panel['target_end_utc'] = (panel['date'] + pd.Timedelta(days=2)).dt.strftime('%Y-%m-%dT00:00:00Z')
    panel['fire_obs_cutoff_utc'] = (panel['date'] - pd.Timedelta(hours=FIRE_LATENCY_H)) \
        .dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    panel['weather_valid_through_utc'] = (panel['date'] - pd.Timedelta(days=lag-1)) \
        .dt.strftime('%Y-%m-%dT00:00:00Z')

    # ---- eligibility and QC
    panel['eligible_new_activity'] = panel['own_known_7d'].eq(0)
    panel['label_status'] = 'ok'
    panel['eligibility_uncertain'] = False
    if day_status is not None:
        st = day_status.assign(date=pd.to_datetime(day_status['date'])).set_index('date')['status']
        panel['target_day_firms_status'] = (panel['date'] + pd.Timedelta(days=1)).map(st).fillna('missing').to_numpy()
        bad_days = set(st[st != 'ok'].index)
        win = pd.Series(False, index=panel.index)
        for k in range(ELIGIBILITY_DAYS + 1):
            win |= (panel['date'] - pd.Timedelta(days=k)).isin(bad_days)
        panel['eligibility_uncertain'] = win.to_numpy()
        panel.loc[panel['target_day_firms_status'] != 'ok', 'label_status'] = 'firms_target_day_not_ok'
    else:
        panel['target_day_firms_status'] = 'not_checked'
    panel['target_cloud_obscured_proxy'] = panel['target_cloud_overpass_pct'] >= CLOUD_OBSCURED_PCT
    if cells_meta is not None:
        panel = panel.merge(cells_meta[['cell_id', 'cell_global_id', 'land_fraction', 'cell_area_km2']],
                            on='cell_id', how='left')

    # Missing inputs are never imputed as zero here: rows lacking any feature are dropped
    # (the model pipeline also refuses NaN-filled weather by construction).
    panel = panel.dropna(subset=FEATURES)
    if predict_only:
        panel['target_nextday_detection'] = np.nan
        panel['target_nextday_count'] = np.nan
        panel['target_nextday_frp'] = np.nan
        panel['label_status'] = 'future'
    else:
        panel = panel[panel['target_nextday_count'].notna()].copy()
        panel['target_nextday_detection'] = (panel['target_nextday_count'] > 0).astype(float)
        notok = panel['label_status'] != 'ok'
        panel.loc[notok, ['target_nextday_detection', 'target_nextday_count', 'target_nextday_frp']] = np.nan
    if not keep_ineligible:
        panel = panel[panel['eligible_new_activity'] & ~panel['eligibility_uncertain']].copy()

    id_cols = ['cell_id'] + (['cell_global_id'] if 'cell_global_id' in panel else []) + \
        ['issue_date', 'target_date', 'issue_time_utc', 'target_start_utc', 'target_end_utc',
         'latitude', 'longitude'] + (['land_fraction', 'cell_area_km2'] if 'land_fraction' in panel else [])
    qc = ['eligible_new_activity', 'eligibility_uncertain', 'label_status', 'target_day_firms_status',
          'target_cloud_overpass_pct', 'target_cloud_obscured_proxy', 'detection_on_issue_day',
          'fire_obs_cutoff_utc', 'weather_valid_through_utc']
    targets = ['target_nextday_detection', 'target_nextday_count', 'target_nextday_frp']
    out = panel[id_cols + [c for c in FEATURES if c not in ('latitude', 'longitude')] + qc + targets]
    return out.sort_values(['issue_date', 'cell_id']).reset_index(drop=True)
