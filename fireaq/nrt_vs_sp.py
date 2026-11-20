"""Compare NOAA-20 near-real-time (NRT) FIRMS detections with archived standard-processing (SP) detections.

Measured for the SAME days and area:
* coverage: detections per UTC day and overpass (day/night) in each product; days missing in either;
* matching: one-to-one nearest pairs within MAX_DT minutes and MAX_DIST_M metres (same overpass, ~2 VIIRS I-band pixels);
* timing: acquisition-time differences of matched pairs; geolocation offsets (m, east/north); FRP ratio;
* confidence: SP-vs-NRT confidence transition matrix; unmatched detections by confidence;
* label impact: agreement of the v0.3 cell-day label (SP: type 0; NRT: static-source mask) per region.

Data access (no result is produced without real data):
* ``api``   : FIRMS area API with FIRMS_MAP_KEY (VIIRS_NOAA20_NRT vs VIIRS_NOAA20_SP over their overlapping dates);
* ``files`` : any NRT CSV(s) + SP CSV(s) you supply (e.g. LANCE daily NRT files via an Earthdata token);
* ``store`` : the daily job's accumulated keyless NRT store vs the FIRMS yearly SP archive once it is published.
"""
from __future__ import annotations

import io
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .core import Grid, clean_firms
from .regions import REGIONS, STEP
from .sources import FIRMS_API, SourceError, _get, fetch_firms

MAX_DT_MIN = 10
MAX_DIST_M = 750.0
EARTH_M = 6_371_008.8
CONF = {'l': 'low', 'n': 'nominal', 'h': 'high', 'low': 'low', 'nominal': 'nominal', 'high': 'high'}


def normalise(df: pd.DataFrame, product: str) -> pd.DataFrame:
    x = df.copy()
    x['acq_time'] = x['acq_time'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
    x['acq_utc'] = pd.to_datetime(x['acq_date'].astype(str) + ' ' + x['acq_time'].str[:2] + ':' + x['acq_time'].str[2:], utc=True)
    x['conf'] = x['confidence'].astype(str).str.lower().map(CONF).fillna('unknown')
    x['latitude'], x['longitude'] = pd.to_numeric(x['latitude']), pd.to_numeric(x['longitude'])
    x['frp'] = pd.to_numeric(x['frp'], errors='coerce')
    x['product'] = product
    keep = ['latitude', 'longitude', 'acq_utc', 'conf', 'frp', 'product'] + \
           [c for c in ['daynight', 'type', 'version', 'confidence', 'acq_date', 'acq_time'] if c in x]
    return x[keep].drop_duplicates(['latitude', 'longitude', 'acq_utc']).reset_index(drop=True)


def match(sp: pd.DataFrame, nrt: pd.DataFrame, max_dt_min=MAX_DT_MIN, max_dist_m=MAX_DIST_M) -> pd.DataFrame:
    """Greedy one-to-one matching by distance among candidates within the time window."""
    from sklearn.neighbors import BallTree
    if sp.empty or nrt.empty:
        return pd.DataFrame(columns=['i_sp', 'i_nrt', 'dist_m', 'dt_min'])
    tree = BallTree(np.radians(nrt[['latitude', 'longitude']].to_numpy()), metric='haversine')
    ind, dist = tree.query_radius(np.radians(sp[['latitude', 'longitude']].to_numpy()), r=max_dist_m / EARTH_M,
                                  return_distance=True)
    cand = []
    t_sp, t_nrt = sp['acq_utc'].to_numpy(), nrt['acq_utc'].to_numpy()
    for i, (js, ds) in enumerate(zip(ind, dist)):
        for j, d in zip(js, ds):
            dt = (t_nrt[j] - t_sp[i]) / np.timedelta64(1, 'm')
            if abs(dt) <= max_dt_min:
                cand.append((d * EARTH_M, i, j, dt))
    cand.sort()
    used_i, used_j, rows = set(), set(), []
    for d, i, j, dt in cand:
        if i not in used_i and j not in used_j:
            used_i.add(i), used_j.add(j)
            rows.append({'i_sp': i, 'i_nrt': j, 'dist_m': d, 'dt_min': dt})
    return pd.DataFrame(rows)


def _offsets(sp, nrt, m):
    la1, lo1 = np.radians(sp.loc[m.i_sp, 'latitude'].to_numpy()), np.radians(sp.loc[m.i_sp, 'longitude'].to_numpy())
    la2, lo2 = np.radians(nrt.loc[m.i_nrt, 'latitude'].to_numpy()), np.radians(nrt.loc[m.i_nrt, 'longitude'].to_numpy())
    north = (la2 - la1) * EARTH_M
    east = (lo2 - lo1) * np.cos((la1 + la2) / 2) * EARTH_M
    return east, north


def compare(sp_raw: pd.DataFrame, nrt_raw: pd.DataFrame, start: str, end: str,
            regions: list | None = None, static_mask: set | None = None) -> dict:
    regions = regions or list(REGIONS)
    s, e = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1)
    out = {'window_utc': [str(start), str(end)], 'match_rule': f'|dt|<={MAX_DT_MIN} min, distance<={MAX_DIST_M} m, one-to-one',
           'regions': {}}
    allm = []
    for reg in regions:
        w, so, ea, no = REGIONS[reg]['bbox']

        def clip(df):
            d = df[df.latitude.between(so, no) & df.longitude.between(w, ea) & (df.acq_utc >= s) & (df.acq_utc < e)]
            return d.reset_index(drop=True)
        sp, nrt = clip(normalise(sp_raw, 'SP')), clip(normalise(nrt_raw, 'NRT'))
        m = match(sp, nrt)
        r = {'sp_detections': int(len(sp)), 'nrt_detections': int(len(nrt)), 'matched': int(len(m)),
             'sp_matched_share': float(len(m) / len(sp)) if len(sp) else None,
             'nrt_matched_share': float(len(m) / len(nrt)) if len(nrt) else None}
        days = pd.date_range(start, end).strftime('%Y-%m-%d')
        cov = pd.DataFrame({'day': days})
        for name, d in [('sp', sp), ('nrt', nrt)]:
            cov[f'{name}_n'] = cov['day'].map(d['acq_utc'].dt.strftime('%Y-%m-%d').value_counts()).fillna(0).astype(int)
        r['days_with_sp_but_no_nrt'] = cov.loc[(cov.sp_n > 0) & (cov.nrt_n == 0), 'day'].tolist()
        r['days_with_nrt_but_no_sp'] = cov.loc[(cov.nrt_n > 0) & (cov.sp_n == 0), 'day'].tolist()
        if 'daynight' in sp and 'daynight' in nrt:
            r['by_overpass'] = {dn: {'sp': int((sp.daynight == dn).sum()), 'nrt': int((nrt.daynight == dn).sum())}
                                for dn in ['D', 'N']}
        if len(m):
            east, north = _offsets(sp, nrt, m)
            r['timing_dt_min'] = {'mean': float(m.dt_min.mean()), 'share_identical_minute': float((m.dt_min == 0).mean()),
                                  'p05': float(m.dt_min.quantile(.05)), 'p95': float(m.dt_min.quantile(.95))}
            r['geolocation_m'] = {'median': float(m.dist_m.median()), 'mean': float(m.dist_m.mean()),
                                  'p95': float(m.dist_m.quantile(.95)), 'mean_east': float(east.mean()),
                                  'mean_north': float(north.mean()), 'share_exact_position': float((m.dist_m < 1).mean())}
            ct = pd.crosstab(sp.loc[m.i_sp, 'conf'].to_numpy(), nrt.loc[m.i_nrt, 'conf'].to_numpy())
            r['confidence_matrix_sp_rows_nrt_cols'] = {i: {j: int(v) for j, v in row.items()} for i, row in ct.iterrows()}
            f1, f2 = sp.loc[m.i_sp, 'frp'].to_numpy(), nrt.loc[m.i_nrt, 'frp'].to_numpy()
            ok = (f1 > 0) & (f2 > 0)
            r['frp_ratio_nrt_over_sp_median'] = float(np.median(f2[ok] / f1[ok])) if ok.any() else None
        um_sp = sp.drop(index=m.i_sp if len(m) else [])
        um_nrt = nrt.drop(index=m.i_nrt if len(m) else [])
        r['unmatched_sp_by_conf'] = um_sp['conf'].value_counts().to_dict()
        r['unmatched_nrt_by_conf'] = um_nrt['conf'].value_counts().to_dict()
        # label impact on the v0.3 grid
        g = Grid(*REGIONS[reg]['bbox'], STEP)
        lab_sp = clean_firms(sp_raw.assign(acq_time=sp_raw['acq_time'].astype(str)), g)
        lab_nrt = clean_firms(nrt_raw.drop(columns=[c for c in ['type'] if c in nrt_raw]).assign(
            acq_time=nrt_raw['acq_time'].astype(str)), g, static_mask=static_mask)
        keys = lambda d: set(zip(d['cell_id'], d['observed_date'].dt.strftime('%Y-%m-%d')))  # noqa: E731
        win = set(days)
        a = {k for k in keys(lab_sp) if k[1] in win}
        b = {k for k in keys(lab_nrt) if k[1] in win}
        r['cell_day_labels'] = {'both_positive': len(a & b), 'sp_only': len(a - b), 'nrt_only': len(b - a),
                                'note': 'SP filtered by type==0; NRT (no type) by the static-source mask, as in live use'}
        out['regions'][reg] = r
        if len(m):
            allm.append(m.assign(region=reg))
    if allm:
        mm = pd.concat(allm)
        out['pooled'] = {'matched': int(len(mm)), 'geolocation_median_m': float(mm.dist_m.median()),
                         'geolocation_p95_m': float(mm.dist_m.quantile(.95)),
                         'dt_share_identical_minute': float((mm.dt_min == 0).mean())}
    tot_sp = sum(v['sp_detections'] for v in out['regions'].values())
    tot_nrt = sum(v['nrt_detections'] for v in out['regions'].values())
    tot_m = sum(v['matched'] for v in out['regions'].values())
    out['totals'] = {'sp': tot_sp, 'nrt': tot_nrt, 'matched': tot_m,
                     'sp_matched_share': tot_m / tot_sp if tot_sp else None,
                     'nrt_matched_share': tot_m / tot_nrt if tot_nrt else None}
    return out


def api_overlap(key: str) -> tuple[date, date]:
    """Dates where FIRMS serves both VIIRS_NOAA20_NRT and VIIRS_NOAA20_SP (data_availability endpoint)."""
    r = _get(f'{FIRMS_API}/data_availability/csv/{key}/ALL', secret=key)
    if 'Invalid' in r.text[:100]:
        raise SourceError('FIRMS rejected the MAP_KEY')
    av = pd.read_csv(io.StringIO(r.text)).set_index('data_id')
    a = max(pd.Timestamp(av.loc['VIIRS_NOAA20_NRT', 'min_date']), pd.Timestamp(av.loc['VIIRS_NOAA20_SP', 'min_date']))
    b = min(pd.Timestamp(av.loc['VIIRS_NOAA20_NRT', 'max_date']), pd.Timestamp(av.loc['VIIRS_NOAA20_SP', 'max_date']))
    if a > b:
        raise SourceError(f'No NRT/SP overlap in FIRMS availability: NRT from {av.loc["VIIRS_NOAA20_NRT", "min_date"]}, '
                          f'SP to {av.loc["VIIRS_NOAA20_SP", "max_date"]}')
    return a.date(), b.date()


def run(mode: str, out: Path, key: str = '', nrt_files=(), sp_files=(), start: str | None = None,
        end: str | None = None, max_days: int = 31, static_mask_path: Path | None = None, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mask = set(json.loads(Path(static_mask_path).read_text())) if static_mask_path and Path(static_mask_path).exists() else None
    status = {'mode': mode, 'checked_at_utc': pd.Timestamp.now(tz='UTC').isoformat()}
    if mode == 'api':
        if not key:
            status.update(state='BLOCKED', reason='FIRMS_MAP_KEY not set. NRT/SP overlap is only reachable keylessly once '
                          'the 2026 yearly SP archive is published (use --mode store then).',
                          unblock='export FIRMS_MAP_KEY=<key from https://firms.modaps.eosdis.nasa.gov/api/map_key/> && '
                                  'python -m fireaq nrt-vs-sp --mode api')
            (out / 'nrt_vs_sp_status.json').write_text(json.dumps(status, indent=2) + '\n')
            return status
        a, b = api_overlap(key)
        if start:
            a = max(a, date.fromisoformat(start))
        b = min(b, a + timedelta(days=max_days - 1))
        raw, manifest = out / 'raw', []
        sp = pd.concat([fetch_firms(key, REGIONS[r]['bbox'], a, b, 'VIIRS_NOAA20_SP', raw, manifest) for r in REGIONS])
        nrt = pd.concat([fetch_firms(key, REGIONS[r]['bbox'], a, b, 'VIIRS_NOAA20_NRT', raw, manifest) for r in REGIONS])
        start, end = str(a), str(b)
    elif mode in ('files', 'store'):
        if not nrt_files or not sp_files:
            status.update(state='BLOCKED', reason='need both NRT and SP inputs for overlapping dates')
            (out / 'nrt_vs_sp_status.json').write_text(json.dumps(status, indent=2) + '\n')
            return status
        rd = lambda f: pd.read_parquet(f) if str(f).endswith('.parquet') else pd.read_csv(f, dtype={'acq_time': str})  # noqa
        nrt = pd.concat([rd(f) for f in nrt_files], ignore_index=True)
        sp = pd.concat([rd(f) for f in sp_files], ignore_index=True)
        n_days = set(pd.to_datetime(nrt['acq_date']).dt.strftime('%Y-%m-%d'))
        s_days = set(pd.to_datetime(sp['acq_date']).dt.strftime('%Y-%m-%d'))
        both = sorted(n_days & s_days)
        if not both:
            status.update(state='BLOCKED', reason=f'no overlapping dates (NRT {min(n_days)}..{max(n_days)}, '
                                                  f'SP {min(s_days)}..{max(s_days)})')
            (out / 'nrt_vs_sp_status.json').write_text(json.dumps(status, indent=2) + '\n')
            return status
        start, end = start or both[0], end or both[-1]
    else:
        raise ValueError(mode)
    res = compare(sp, nrt, start, end, static_mask=mask)
    res['state'] = 'MEASURED'
    (out / 'nrt_vs_sp_v3.json').write_text(json.dumps(res, indent=2, default=str) + '\n')
    (out / 'nrt_vs_sp_status.json').write_text(json.dumps({**status, 'state': 'MEASURED', 'window': [start, end]}, indent=2) + '\n')
    log(f"MEASURED {start}..{end}: {res['totals']}")
    return res
