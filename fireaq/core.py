"""Create an *observed-detection* research dataset with strictly past-only features.

This project DOES NOT infer ignition, fire spread, or true wildfire absence.
Target is qualifying satellite detection in grid cell on calendar day d+1,
conditional on no *known* detection in cell in the preceding seven issue days.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable
import numpy as np
import pandas as pd

FIRE_LATENCY_H = 3  # planning assumption for NRT; science products are retrospective
WEATHER_LAG_DAYS = 2
FEATURES = ['latitude', 'longitude', 'doy_sin', 'doy_cos',
            'temperature_lag2', 'humidity_lag2', 'wind_lag2',
            'rain_3d_lag2', 'rain_7d_lag2',
            'neighbor_fire_3d', 'neighbor_fire_7d', 'neighbor_frp_3d']


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

    def cells(self):
        return pd.DataFrame([{
            'cell_id': f'r{y}_c{x}', 'ix': x, 'iy': y,
            'latitude': self.south + (y+0.5)*self.step,
            'longitude': self.west + (x+0.5)*self.step,
        } for y in range(self.ny) for x in range(self.nx)])

    def assign(self, lat: pd.Series, lon: pd.Series):
        x = np.floor((pd.to_numeric(lon, errors='coerce').to_numpy()-self.west)/self.step).astype(float)
        y = np.floor((pd.to_numeric(lat, errors='coerce').to_numpy()-self.south)/self.step).astype(float)
        good = (x >= 0)&(x<self.nx)&(y>=0)&(y<self.ny)&np.isfinite(x)&np.isfinite(y)
        names = np.full(len(x), None, dtype=object)
        names[good] = [f'r{int(yy)}_c{int(xx)}' for xx,yy in zip(x[good],y[good])]
        return names


def clean_firms(df: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    """Parse VIIRS FIRMS CSV, retaining nominal/high presumed vegetation detections.
    Product 'type' is sometimes omitted in NRT, so absence is explicitly recorded.
    """
    base = ['latitude', 'longitude', 'acq_date', 'acq_time', 'confidence', 'frp']
    absent = set(base) - set(df.columns)
    if absent:
        raise ValueError(f'FIRMS missing required columns: {absent}')
    x = df.copy()
    x['latitude'] = pd.to_numeric(x['latitude'], errors='coerce')
    x['longitude'] = pd.to_numeric(x['longitude'], errors='coerce')
    x['frp'] = pd.to_numeric(x['frp'], errors='coerce')
    x = x[x['confidence'].astype(str).str.lower().isin(['n','h','nominal','high'])]
    if 'type' in x:
        x = x[pd.to_numeric(x['type'], errors='coerce') == 0]
    t = x['acq_time'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
    x['observed_utc'] = pd.to_datetime(x['acq_date'].astype(str)+' '+t.str[:2]+':'+t.str[2:], utc=True, errors='coerce')
    x = x[x['observed_utc'].notna() & x['frp'].ge(0)]
    x['cell_id'] = grid.assign(x['latitude'],x['longitude'])
    x = x[x['cell_id'].notna()].copy()
    keys = ['cell_id', 'observed_utc', 'latitude', 'longitude']
    if 'source_product' in x:
        keys.append('source_product')
    x = x.drop_duplicates(keys)
    x['observed_date'] = x['observed_utc'].dt.tz_localize(None).dt.normalize()
    # First UTC-midnight issue time after data would have become available
    x['first_usable_issue'] = (x['observed_utc'] + pd.Timedelta(hours=FIRE_LATENCY_H)).dt.ceil('D').dt.tz_localize(None)
    return x[['cell_id','observed_utc','observed_date','first_usable_issue','frp']].reset_index(drop=True)


def make_dataset(fires_raw: pd.DataFrame, weather: pd.DataFrame, grid: Grid,
                 start: str, end: str, *, predict_only=False) -> pd.DataFrame:
    """Create cell × issue-day records; target uses NEXT day, predictors <= issue midnight.

    A complete NASA acquisition-window download must be established BEFORE invoking
    this on real data. Missing detections alone cannot identify satellite coverage.
    """
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError('start after end')
    cells = grid.cells()
    dates = pd.date_range(start_ts - pd.Timedelta(days=10), end_ts + pd.Timedelta(days=1))
    all_idx = pd.MultiIndex.from_product([cells['cell_id'], dates], names=['cell_id','date'])
    panel = pd.DataFrame(index=all_idx)
    if fires_raw.empty:
        raise ValueError('FIRMS data is empty: refusing to convert missing acquisitions into zero observations')
    fires = clean_firms(fires_raw, grid)
    if fires.empty:
        raise ValueError('No qualifying detections in the downloaded window: choose a broader/higher-activity pilot')
    known = fires.groupby(['cell_id','first_usable_issue']).agg(
        known_count=('frp','size'), known_frp=('frp','sum'))
    known.index = known.index.set_names(['cell_id','date'])
    observed = fires.groupby(['cell_id','observed_date']).size().rename('observed_count')
    observed.index = observed.index.set_names(['cell_id','date'])
    panel = panel.join(known).join(observed).fillna({'known_count':0,'known_frp':0,'observed_count':0})
    # Keep complete series per cell so lags never bridge missing dates.
    panel['known_7d'] = panel.groupby(level='cell_id')['known_count'].transform(lambda s: s.rolling(7,min_periods=7).sum())
    panel['known_3d'] = panel.groupby(level='cell_id')['known_count'].transform(lambda s: s.rolling(3,min_periods=3).sum())
    panel['frp_3d'] = panel.groupby(level='cell_id')['known_frp'].transform(lambda s: s.rolling(3,min_periods=3).sum())
    cids = list(cells['cell_id'])
    index_lookup = {(int(row.ix),int(row.iy)): row.cell_id for row in cells.itertuples()}
    for col, outcol in [('known_3d','neighbor_fire_3d'), ('known_7d','neighbor_fire_7d'),('frp_3d','neighbor_frp_3d')]:
        # Use the same date window for every cell. No future values in these sums.
        arr = {cid: panel.xs(cid,level='cell_id')[col].to_numpy(dtype=float) for cid in cids}
        chunks=[]
        for cell in cells.itertuples():
            ns = [index_lookup.get((cell.ix+dx,cell.iy+dy))
                  for dx in (-1,0,1) for dy in (-1,0,1) if (dx,dy)!=(0,0)]
            ns = [n for n in ns if n is not None]
            chunks.append(np.sum([arr[n] for n in ns],axis=0) if ns else np.zeros(len(dates)))
        panel[outcol] = np.concatenate(chunks)
    wxcols = {'temp_c':'temperature_lag2','rh_pct':'humidity_lag2','wind_ms':'wind_lag2'}
    must = {'cell_id','date','temp_c','rh_pct','precip_mm','wind_ms','n_hours'}
    if not must.issubset(weather.columns):
        raise ValueError(f'weather missing columns: {must - set(weather.columns)}')
    wx=weather.copy()
    wx['date']=pd.to_datetime(wx['date']).dt.normalize()
    wx=wx.drop_duplicates(['cell_id','date'],keep=False).set_index(['cell_id','date']).reindex(all_idx)
    wx.loc[wx['n_hours'] != 24, ['temp_c','rh_pct','precip_mm','wind_ms']] = np.nan
    for src,dst in wxcols.items():
        panel[dst]=wx.groupby(level='cell_id')[src].shift(WEATHER_LAG_DAYS)
    for days in (3,7):
        daily_rain = wx.groupby(level='cell_id')['precip_mm'].transform(
            lambda s:s.rolling(days,min_periods=days).sum())
        panel[f'rain_{days}d_lag2']=daily_rain.groupby(level='cell_id').shift(WEATHER_LAG_DAYS)
    panel['nextday_observed_count'] = panel.groupby(level='cell_id')['observed_count'].shift(-1)
    panel['target_nextday_detection'] = (panel['nextday_observed_count'] > 0).astype(int)
    panel['eligible_new_activity'] = panel['known_7d'].eq(0)
    panel = panel.reset_index().merge(cells[['cell_id','latitude','longitude']], on='cell_id', how='left')
    panel = panel[panel['date'].between(start_ts,end_ts)].copy()
    doy=panel['date'].dt.dayofyear
    panel['doy_sin'] = np.sin(2*np.pi*doy/365.25)
    panel['doy_cos'] = np.cos(2*np.pi*doy/365.25)
    panel['target_date'] = (panel['date']+pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    panel['issue_date'] = panel['date'].dt.strftime('%Y-%m-%d')
    panel = panel[panel['eligible_new_activity']].copy()
    # No missing input: do not silently fill unknown weather with zeros.
    panel = panel.dropna(subset=FEATURES)
    if not predict_only:
        panel = panel[panel['nextday_observed_count'].notna()]
    else:
        panel['target_nextday_detection'] = pd.NA
        panel['nextday_observed_count'] = pd.NA
    return panel[['cell_id','issue_date','target_date','latitude','longitude',
                  'eligible_new_activity',* [c for c in FEATURES if c not in ('latitude','longitude')],
                  'target_nextday_detection']].reset_index(drop=True)
