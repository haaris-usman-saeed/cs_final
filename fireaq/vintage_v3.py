"""POWER weather-vintage check for v0.3: MERRA-2 (training) vs GEOS-IT (live), bridged through ERA5.

NASA POWER serves MERRA-2 up to 2026-10-01 and GEOS-IT from 2026-10-02 (probed 2026-11-20); the streams never overlap.
Design (double difference, same cells, same calendar window):
    bias_M = POWER(MERRA-2) - ERA5   for Oct 2 - Nov 14 of 2024 and of 2025
    bias_G = POWER(GEOS-IT) - ERA5   for Oct 2 - Nov 14 of 2026
    shift  = bias_G - mean(bias_M)               (per cell, then summarised with a cell bootstrap)
    null   = bias_M(2025) - bias_M(2024)          (same calculation with no stream change: year-to-year noise)
Soil wetness (fraction of saturation) and ERA5 soil moisture (m3/m3) differ in units, so for gwettop/gwetroot the
reference is a per-cell linear map POWER ~ a + b*ERA5 fitted on the MERRA-2 years.
The measured shifts are then pushed through the frozen v0.3 model on the already-run test rows (sensitivity only;
nothing is refit or re-selected).
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .features_v3 import TARGET
from .regions import REGIONS
from .sources import ARCHIVE_API, SourceError, _get, cached_entry, fetch_power_daily, save_raw, write_manifest

WINDOWS = {'merra2_2024': (date(2024, 10, 2), date(2024, 11, 14)),
           'merra2_2025': (date(2025, 10, 2), date(2025, 11, 14)),
           'geosit_2026': (date(2026, 10, 2), date(2026, 11, 14))}
SWITCH_DATE = '2026-10-02'
PAIRS = {'t2m': 'era5_t2m', 't2m_max': 'era5_t2m_max', 'rh2m': 'era5_rh', 'ws10m': 'era5_ws',
         'ws10m_max': 'era5_ws_max', 'precip_mm': 'era5_precip', 'gwettop': 'era5_sm_top', 'gwetroot': 'era5_sm_root'}
MAPPED = {'gwettop', 'gwetroot'}


def fetch_era5_daily(cell_key: str, lat: float, lon: float, start: date, end: date, rawdir: Path, manifest: list,
                     pace: float = 1.0) -> pd.DataFrame:
    fname = f"{cell_key.replace(':', '_')}_ERA5daily_{start:%Y%m%d}_{end:%Y%m%d}.json"
    dst = rawdir / fname
    hit = cached_entry(dst)
    if hit:
        content = dst.read_bytes()
        manifest.append(hit)
    else:
        params = {'latitude': lat, 'longitude': lon, 'start_date': str(start), 'end_date': str(end), 'models': 'era5',
                  'timezone': 'GMT', 'wind_speed_unit': 'ms', 'cell_selection': 'nearest',
                  'daily': 'temperature_2m_mean,temperature_2m_max,relative_humidity_2m_mean,precipitation_sum,'
                           'wind_speed_10m_mean,wind_speed_10m_max',
                  'hourly': 'soil_moisture_0_to_7cm,soil_moisture_28_to_100cm'}
        r = _get(ARCHIVE_API, params=params, timeout=(15, 60))
        content = r.content
        if json.loads(content).get('error'):
            raise SourceError(f'ERA5 error {cell_key}: {json.loads(content).get("reason")}')
        save_raw(rawdir, fname, content, 'Open-Meteo ERA5 (vintage bridge)', r.url, manifest,
                 product='ECMWF ERA5 via Open-Meteo archive, daily UTC + hourly soil moisture', cell_key=cell_key)
        time.sleep(pace)
    p = json.loads(content)
    d = pd.DataFrame(p['daily'])
    h = pd.DataFrame(p['hourly'])
    h['date'] = h['time'].str[:10]
    hg = h.groupby('date')
    soil = pd.DataFrame({'era5_sm_top': hg['soil_moisture_0_to_7cm'].mean(),
                         'era5_sm_root': hg['soil_moisture_28_to_100cm'].mean(),
                         'n': hg['soil_moisture_0_to_7cm'].count()})
    soil.loc[soil['n'] < 24, ['era5_sm_top', 'era5_sm_root']] = np.nan
    out = pd.DataFrame({'date': d['time'], 'era5_t2m': d['temperature_2m_mean'], 'era5_t2m_max': d['temperature_2m_max'],
                        'era5_rh': d['relative_humidity_2m_mean'], 'era5_precip': d['precipitation_sum'],
                        'era5_ws': d['wind_speed_10m_mean'], 'era5_ws_max': d['wind_speed_10m_max']})
    return out.merge(soil[['era5_sm_top', 'era5_sm_root']].reset_index(), on='date', how='left')


def _boot_mean(x, n=2000, seed=17):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _vpd(tmax, rh):
    es = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    return es * (1 - np.clip(rh, 0, 100) / 100.0)


def compute_shifts(df: pd.DataFrame, log=print):
    """Per-variable GEOS-IT minus MERRA-2 shift (double difference vs ERA5) and a 2025-vs-2024 null."""
    # per-cell bias in each window (identity variables) or residual from MERRA-fitted map (soil)
    per_cell = []
    for ck, g in df.groupby('cell_key'):
        rec = {'cell_key': ck, 'region': g['region'].iloc[0]}
        for v, e in PAIRS.items():
            if v in MAPPED:
                fit = g[g.window.str.startswith('merra2')].dropna(subset=[v, e])
                if len(fit) < 20 or fit[e].std() == 0:
                    continue
                b, a = np.polyfit(fit[e], fit[v], 1)
                resid = g[v] - (a + b * g[e])
                for w in WINDOWS:
                    rec[f'{v}|{w}'] = float(resid[g.window == w].mean())
                # year-to-year null for mapped vars: fit on 2024 only, residual in 2025
                f24 = g[g.window == 'merra2_2024'].dropna(subset=[v, e])
                if len(f24) > 10 and f24[e].std() > 0:
                    b2, a2 = np.polyfit(f24[e], f24[v], 1)
                    r = g[v] - (a2 + b2 * g[e])
                    rec[f'{v}|null_fit2024'] = float(r[g.window == 'merra2_2025'].mean())
                    rec[f'{v}|shift_fit2024'] = float(r[g.window == 'geosit_2026'].mean())
            else:
                for w in WINDOWS:
                    gw = g[g.window == w]
                    rec[f'{v}|{w}'] = float((gw[v] - gw[e]).mean())
                    rec[f'{v}|corr|{w}'] = float(gw[[v, e]].corr().iloc[0, 1]) if len(gw) > 5 else np.nan
        per_cell.append(rec)
    pc = pd.DataFrame(per_cell)
    summary = {}
    for v in PAIRS:
        if f'{v}|geosit_2026' not in pc:
            continue
        if v in MAPPED:
            shift = pc[f'{v}|shift_fit2024'] if f'{v}|shift_fit2024' in pc else pc[f'{v}|geosit_2026']
            null = pc[f'{v}|null_fit2024']
            note = 'residual from per-cell linear map POWER~ERA5 fitted on MERRA-2 2024 (units: POWER soil wetness)'
        else:
            mb = (pc[f'{v}|merra2_2024'] + pc[f'{v}|merra2_2025']) / 2
            shift = pc[f'{v}|geosit_2026'] - mb
            null = pc[f'{v}|merra2_2025'] - pc[f'{v}|merra2_2024']
            note = 'double difference vs ERA5 (POWER units)'
        s_mean, s_ci = _boot_mean(shift)
        n_mean, n_ci = _boot_mean(null)
        by_region = {r: round(float(shift[pc.region == r].mean()), 4) for r in REGIONS}
        corr = ({w: round(float(pc[f'{v}|corr|{w}'].mean()), 3) for w in WINDOWS} if f'{v}|corr|merra2_2024' in pc else None)
        summary[v] = {'shift_mean': s_mean, 'shift_ci95_cells': s_ci, 'null_mean_2025_minus_2024': n_mean,
                      'null_ci95_cells': n_ci, 'shift_abs_exceeds_null_abs': abs(s_mean) > abs(n_mean),
                      'shift_by_region': by_region, 'mean_corr_with_era5_by_window': corr, 'method': note}
        log(f"{v:10s} shift {s_mean:+.3f} {np.round(s_ci, 3)}  null {n_mean:+.3f} {np.round(n_ci, 3)}")
    # precipitation also as a ratio (multiplicative shift for rain sums)
    tot = df.groupby('window')[['precip_mm', 'era5_precip']].sum()
    ratio = {w: float(tot.loc[w, 'precip_mm'] / tot.loc[w, 'era5_precip']) for w in tot.index}

    return summary, ratio, pc


def run_power_vintage_check(root: Path = Path('data/v03'), outdir: Path = Path('output_v3'),
                            cells_per_region: int = 16, log=print) -> dict:
    root, outdir = Path(root), Path(outdir)
    raw = root / 'raw' / 'vintage_power'
    manifest, rows = [], []
    for reg in REGIONS:
        cells = pd.read_csv(root / 'interim' / reg / 'cells.csv')
        cells = cells[cells.included_land_cell & (cells.iy % 2 == 0) & (cells.ix % 2 == 0)].head(cells_per_region)
        for c in cells.itertuples():
            pw = fetch_power_daily(c.cell_id, c.latitude, c.longitude, WINDOWS['merra2_2024'][0],
                                   WINDOWS['geosit_2026'][1], raw / reg, manifest)
            for win, (a, b) in WINDOWS.items():
                e5 = fetch_era5_daily(c.cell_key, c.latitude, c.longitude, a, b, raw / reg / 'era5', manifest)
                m = pw[(pw.date >= str(a)) & (pw.date <= str(b))].merge(e5, on='date')
                m['window'], m['region'], m['cell_key'] = win, reg, c.cell_key
                rows.append(m)
        log(f'{reg}: {len(cells)} cells fetched')
    write_manifest(outdir / 'vintage_power_manifest.json', manifest)
    df = pd.concat(rows, ignore_index=True)
    # stream check: every geosit_2026 day must really be GEOS-IT and every merra day MERRA-2
    src_ok = {w: sorted(df.loc[df.window == w, 'power_sources'].unique().tolist()) for w in WINDOWS}

    summary, ratio, pc = compute_shifts(df, log)

    # ---- propagate through the frozen model (sensitivity on already-tested rows; no refit)
    bundle = joblib.load(outdir / 'model_v3.joblib')
    pred = pd.read_parquet(root / 'dataset' / 'firerisk_open_multiregion_v0.3.0.parquet')
    from .experiment_v3 import assign_roles
    pred = pred[pred.in_primary_analysis & pred[TARGET].notna()]
    pred = pred[assign_roles(pred).isin(['test_time', 'test_space_primary'])].reset_index(drop=True)
    feats = bundle['features']

    def score(X):
        return np.clip(bundle['calibrator'].predict(bundle['model'].predict_proba(X[feats])[:, 1]), 0, 1)
    base = score(pred)
    shifted = pred.copy()
    regs = shifted['region']
    add = {'t2m_lag2': 't2m', 't2m_max_lag2': 't2m_max', 'rh2m_lag2': 'rh2m', 'ws10m_lag2': 'ws10m',
           'ws10m_max_lag2': 'ws10m_max', 'gwettop_lag2': 'gwettop', 'gwetroot_lag2': 'gwetroot'}
    for f, v in add.items():
        if v in summary:
            shifted[f] = shifted[f] + regs.map(summary[v]['shift_by_region']).fillna(summary[v]['shift_mean'])
    shifted['rh2m_lag2'] = shifted['rh2m_lag2'].clip(0, 100)
    shifted['ws10m_lag2'] = shifted['ws10m_lag2'].clip(lower=0)
    shifted['ws10m_max_lag2'] = shifted['ws10m_max_lag2'].clip(lower=0)
    for f in ['gwettop_lag2', 'gwetroot_lag2']:
        shifted[f] = shifted[f].clip(0, 1)
    shifted['vpd_lag2'] = _vpd(shifted['t2m_max_lag2'], shifted['rh2m_lag2'])
    pr = ratio['geosit_2026'] / np.mean([ratio['merra2_2024'], ratio['merra2_2025']])
    for f in ['rain_3d_lag2', 'rain_7d_lag2', 'rain_30d_lag2', 'rain_90d_lag2']:
        shifted[f] = shifted[f] * pr
    p2 = score(shifted)
    y = pred[TARGET].astype(int).to_numpy()
    thr = bundle['threshold']
    prop = {'rows': int(len(pred)), 'precip_ratio_applied': float(pr),
            'mean_abs_prob_change': float(np.mean(np.abs(p2 - base))),
            'p95_abs_prob_change': float(np.percentile(np.abs(p2 - base), 95)),
            'mean_prob_before_after': [float(base.mean()), float(p2.mean())],
            'threshold_decision_agreement': float(np.mean((p2 >= thr) == (base >= thr))),
            'alerts_before_after': [int((base >= thr).sum()), int((p2 >= thr).sum())],
            'ap_before_after': {}, 'note': 'region-mean additive shifts (precip multiplicative) applied to the test rows; '
                                           'dry_days and gwetroot_chg30 left unchanged'}
    for role_name, mask in [('all', np.ones(len(pred), bool))] + [(r, (pred.region == r).to_numpy()) for r in REGIONS]:
        if y[mask].sum():
            prop['ap_before_after'][role_name] = [float(average_precision_score(y[mask], base[mask])),
                                                  float(average_precision_score(y[mask], p2[mask]))]
    res = {'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'power_stream_switch_date': SWITCH_DATE,
           'windows': {k: [str(a), str(b)] for k, (a, b) in WINDOWS.items()},
           'power_sources_request_header': src_ok,
           'per_day_stream_verification': 'POWER reports sources per request, not per day, so the header above lists '
                                          'every stream in the 2024-2026 request. The per-day stream was verified by '
                                          'single-day probes on 2026-11-20: 2026-10-01 MERRA2, 2026-10-02..11-19 GEOSIT, '
                                          'all sampled 2025 months MERRA2.',
           'caveats': ['autumn window only (Oct 2 - Nov 14); shifts may differ in the summer fire season',
                       'mean additive shifts only; GEOS-IT day-to-day correlation with ERA5 is reported per window',
                       'dry_days and gwetroot_chg30 were not perturbed in the sensitivity run'],
           'cells': int(pc['cell_key'].nunique()), 'variable_shift_geosit_minus_merra2': summary,
           'precip_total_ratio_power_over_era5': ratio, 'model_sensitivity_on_test_rows': prop}
    (outdir / 'vintage_power_v3.json').write_text(json.dumps(res, indent=2, default=str) + '\n')
    pc.to_csv(outdir / 'vintage_power_per_cell_v3.csv', index=False)
    return res
