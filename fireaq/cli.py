"""FireRisk Open command line. ``python -m fireaq --help``"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .core import ELIGIBILITY_DAYS, FIRE_LATENCY_H, HISTORY_DAYS, Grid, make_dataset

PILOT_BBOX = (-123.5, 38.5, -119.5, 42.5)


def load_env_key() -> str:
    """FIRMS_MAP_KEY from the environment or a git-ignored ``.env``; never printed."""
    key = os.environ.get('FIRMS_MAP_KEY', '')
    env = Path('.env')
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith('FIRMS_MAP_KEY='):
                key = line.split('=', 1)[1].strip().strip('"').strip("'")
    return key


def parse_bbox(text):
    try:
        coords = tuple(map(float, text.split(',')))
    except ValueError as e:
        raise argparse.ArgumentTypeError('Use --bbox=west,south,east,north') from e
    if len(coords) != 4:
        raise argparse.ArgumentTypeError('bbox needs 4 numbers')
    return coords


def get_grid(bbox, step, max_cells):
    grid = Grid(*bbox, step)
    if len(grid.cells()) > max_cells:
        raise ValueError(f'Grid has {len(grid.cells())} cells; cap is {max_cells}. '
                         'Start with a smaller region, or explicitly raise --max-cells.')
    return grid


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ------------------------------------------------------------------ ingest (MAP_KEY API, legacy layout)
def ingest(args):
    from .sources import fetch_firms, fetch_weather_historical, write_manifest
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = get_grid(args.bbox, args.step, args.max_cells)
    key = load_env_key()
    if not key:
        raise SystemExit('BLOCKED: FIRMS_MAP_KEY is not set. Get one at '
                         'https://firms.modaps.eosdis.nasa.gov/api/map_key/ and `export FIRMS_MAP_KEY=...`, '
                         'or use the keyless `ingest-archive` command.')
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if end < start:
        raise ValueError('end must be >= start')
    if end > datetime.now(timezone.utc).date() - timedelta(days=6):
        raise ValueError('For historical ERA5, end should be at least 6 days old; live mode is separate.')
    sources = args.sources.split(',')
    if any(not s.endswith('_SP') for s in sources):
        raise ValueError('Historical training requires _SP standard products; NRT is for live mode')
    manifest = []
    fire_start, fire_end = start - timedelta(days=45), end + timedelta(days=1)
    fires = pd.concat([fetch_firms(key, args.bbox, fire_start, fire_end, s, out / 'raw', manifest)
                       for s in sources], ignore_index=True)
    if fires.empty:
        raise ValueError('FIRMS returned no rows. No zero-filled dataset will be created.')
    save_csv(fires, out / 'firms.csv')
    weather = [fetch_weather_historical(r.cell_id, r.latitude, r.longitude, fire_start, fire_end,
                                        out / 'raw', manifest) for r in grid.cells().itertuples()]
    save_csv(pd.concat(weather, ignore_index=True), out / 'weather.csv')
    meta = {'mode': 'api', 'bbox': list(args.bbox), 'step': args.step, 'grid_cells': len(grid.cells()),
            'fire_source_products': sources, 'fire_start': str(fire_start), 'fire_end': str(fire_end),
            'issue_start': str(start), 'issue_end': str(end),
            'weather_source': 'Open-Meteo ERA5 reanalysis', 'download_complete': True, 'live': False,
            'raw_manifest_file': 'manifest.json'}
    (out / 'ingest_metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    write_manifest(out / 'manifest.json', manifest)
    print(f'Collected {len(fires):,} FIRMS rows; {len(grid.cells())} weather locations; files in {out}')


def ingest_archive_cmd(args):
    from .ingest import ingest_archive
    grid = get_grid(args.bbox, args.step, args.max_cells)
    rep = ingest_archive(args.bbox, args.step, args.start, args.end, Path(args.out), satellite=args.satellite)
    print(json.dumps({k: rep[k] for k in ['firms_regional_rows_total', 'firms_day_status_counts',
                                          'weather', 'raw_files', 'network_downloads_this_run']},
                     indent=2, default=str))
    return grid


def build(args):
    root = Path(args.input)
    meta = json.loads((root / 'ingest_metadata.json').read_text())
    if meta.get('live'):
        raise ValueError('For live predictions use predict-live, not historical build')
    if not meta.get('download_complete'):
        raise ValueError('Cannot build: incomplete data extraction')
    if meta.get('mode') == 'archive':
        from .dataset import build_release
        rep = build_release(root)
        print(json.dumps(rep['row_accounting'], indent=2))
        return
    grid = Grid(*meta['bbox'], meta['step'])
    data = make_dataset(pd.read_csv(root / 'firms.csv'), pd.read_csv(root / 'weather.csv'), grid,
                        meta['issue_start'], meta['issue_end'])
    if data.empty:
        raise ValueError('No eligible training rows: inspect completeness and study coverage')
    out = Path(args.out or root / 'dataset' / 'dataset.csv')
    save_csv(data, out)
    print(f"Built {len(data):,} eligible cell-days, {int(data['target_nextday_detection'].sum()):,} positives; {out}")


def _load_primary(path):
    df = pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)
    if 'in_primary_analysis' in df:
        df = df[df['in_primary_analysis']].copy()
    return df


def experiment(args):
    from .experiment import run_development, run_regression_development
    df = _load_primary(args.data)
    out = Path(args.out)
    if (out / 'test_results.json').exists():
        raise SystemExit(f'{out}/test_results.json exists: development is closed for this output directory.')
    run_development(df, out)
    if not args.skip_regression:
        run_regression_development(df, out)
    print(f'Validation leaderboard: {out}/validation_leaderboard.csv; champion frozen: {out}/frozen_champion.json')


def evaluate_test(args):
    from .experiment import run_final_test
    res = run_final_test(_load_primary(args.data), Path(args.out))
    c = res['champion_test']
    print(json.dumps({'champion': res['champion_key'], 'test_AP': c['average_precision'],
                      'test_prevalence': c['prevalence'], 'test_ROC_AUC': c['roc_auc'],
                      'test_balanced_accuracy': c['balanced_accuracy'], 'test_brier': c['brier']}, indent=2))


def vintage_check(args):
    from .vintage import run_vintage_check
    res = run_vintage_check(Path(args.input), Path(args.out), _load_primary(args.data))
    print(json.dumps(res['summary'], indent=2, default=str))


# ------------------------------------------------------------------ live
def _update_nrt_store(live_dir: Path, manifest: list, key: str, issue: date) -> pd.DataFrame:
    from .sources import fetch_firms, fetch_firms_nrt_keyless
    store_path = live_dir / 'nrt_store.parquet'
    parts = [pd.read_parquet(store_path)] if store_path.exists() else []
    parts.append(fetch_firms_nrt_keyless(live_dir / 'raw', manifest, satellite='noaa20'))
    if key:   # optional backfill of the 31-day history through the MAP_KEY API
        parts.append(fetch_firms(key, PILOT_BBOX, issue - timedelta(days=HISTORY_DAYS + 3), issue,
                                 'VIIRS_NOAA20_NRT', live_dir / 'raw', manifest))
    store = pd.concat(parts, ignore_index=True)
    for c in ['acq_date', 'acq_time', 'confidence', 'version']:
        if c in store:
            store[c] = store[c].astype(str)
    store['acq_time'] = store['acq_time'].str.replace(r'\.0$', '', regex=True).str.zfill(4)
    store = store.drop_duplicates(['latitude', 'longitude', 'acq_date', 'acq_time'])
    store.to_parquet(store_path, index=False)
    return store


def predict_live(args):
    import joblib
    from .model import score_predictions
    from .sources import fetch_weather_live, write_manifest
    model = Path(args.model)
    if not model.exists():
        raise SystemExit('No trained model. Do not publish predictions from fabricated data.')
    bundle = joblib.load(model)
    meta = bundle['metadata']
    if not meta.get('test_evaluated'):
        raise SystemExit('Model has not completed its one-time held-out test; refusing live use.')
    grid = get_grid(args.bbox, args.step, args.max_cells)
    b = meta['training_region_bounds']
    for c in grid.cells().itertuples():
        if not (b['min_latitude']-1e-6 <= c.latitude <= b['max_latitude']+1e-6
                and b['min_longitude']-1e-6 <= c.longitude <= b['max_longitude']+1e-6):
            raise SystemExit('Live region is outside the model training region; retrain and validate first')
    live = Path(args.out)
    live.mkdir(parents=True, exist_ok=True)
    issue = datetime.now(timezone.utc).date()
    manifest = []
    store = _update_nrt_store(live, manifest, load_env_key(), issue)
    obs = pd.to_datetime(store['acq_date'] + ' ' + store['acq_time'].str[:2] + ':' + store['acq_time'].str[2:], utc=True)
    # Days before the store starts would be read as "no fire", so require full coverage of the longest
    # fire window the frozen model uses (eligibility alone always needs 7 days).
    days_needed = HISTORY_DAYS if 'own_fire_30d' in bundle['features'] else ELIGIBILITY_DAYS
    need_from = pd.Timestamp(issue, tz='UTC') - pd.Timedelta(days=days_needed, hours=FIRE_LATENCY_H)
    status = {'issue_time_utc': f'{issue}T00:00:00Z', 'nrt_store_rows': int(len(store)),
              'nrt_history_start_utc': obs.min().isoformat(), 'history_required_from_utc': need_from.isoformat()}
    if obs.min() > need_from:
        status['state'] = 'BLOCKED_INSUFFICIENT_NRT_HISTORY'
        status['detail'] = (f'Fire features need NRT detections back to {need_from:%Y-%m-%d %H:%MZ} ({days_needed}-day window); the local store '
                            f'starts {obs.min():%Y-%m-%d %H:%MZ}. Without a FIRMS_MAP_KEY the keyless feed only '
                            'covers 7 days. Set FIRMS_MAP_KEY (backfills automatically) or run this daily until the '
                            'store spans 31 days.')
        (live / 'live_status.json').write_text(json.dumps(status, indent=2) + '\n')
        write_manifest(live / 'manifest.json', manifest)
        raise SystemExit(status['detail'])
    mask_path = Path(args.static_mask)
    static_mask = set(json.loads(mask_path.read_text())) if mask_path.exists() else None
    wx = []
    for c in grid.cells().itertuples():
        wx.append(fetch_weather_live(c.cell_id, c.latitude, c.longitude, live / 'raw', manifest))
        time.sleep(0.5)
    wx = pd.concat(wx, ignore_index=True)
    ds = make_dataset(store, wx, grid, str(issue), str(issue), predict_only=True, pad_cells=2,
                      static_mask=static_mask)
    write_manifest(live / 'manifest.json', manifest)
    if ds.empty:
        raise SystemExit('No forecastable cells after eligibility and feature completeness rules')
    scores, mmeta = score_predictions(model, ds)
    obj = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'issue_time_utc': f'{issue}T00:00:00Z', 'target_date': str(issue + timedelta(days=1)),
        'target_window_utc': [f'{issue + timedelta(days=1)}T00:00:00Z', f'{issue + timedelta(days=2)}T00:00:00Z'],
        'risk_definition': 'probability of a qualifying NOAA-20 VIIRS detection in the cell, NOT probability of a wildfire ignition',
        'forecast_status': 'EXPERIMENTAL - UNVALIDATED UNDER LIVE INPUTS - NOT FOR EMERGENCY USE',
        'data_vintages': {'fires': 'NASA FIRMS NOAA-20 NRT (no type field; static-source mask applied)',
                          'weather': 'Open-Meteo forecast API past days (operational models, differs from training ERA5)'},
        'model': mmeta['champion']['key'], 'test_average_precision': mmeta.get('test_average_precision'),
        'test_base_rate': mmeta.get('test_prevalence'), 'cells_scored': len(scores),
        'cells_not_scored_reason': 'cells with a detection known in the last 7 days are excluded by design',
        'predictions': scores.to_dict(orient='records'),
    }
    path = Path(args.predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n')
    status['state'] = 'OK'
    (live / 'live_status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(f'Wrote {len(scores)} experimental cell probabilities to {path}')


def main(argv=None):
    p = argparse.ArgumentParser(description='FireRisk Open - original data + research-grade satellite-detection ML')
    sub = p.add_subparsers(dest='command', required=True)

    def region(sp, out='data/pilot'):
        sp.add_argument('--bbox', type=parse_bbox, default=PILOT_BBOX, help='west,south,east,north')
        sp.add_argument('--step', type=float, default=0.5, help='grid degrees, default 0.5')
        sp.add_argument('--max-cells', type=int, default=64)
        sp.add_argument('--out', default=out)
    a = sub.add_parser('ingest', help='Download historical data through the FIRMS MAP_KEY API')
    region(a)
    a.add_argument('--start', required=True)
    a.add_argument('--end', required=True)
    a.add_argument('--sources', default='VIIRS_NOAA20_SP')
    ar = sub.add_parser('ingest-archive', help='Keyless: FIRMS yearly archive + ERA5 + land mask')
    region(ar, 'data/norcal')
    ar.add_argument('--start', default='2019-01-01')
    ar.add_argument('--end', default='2025-12-30')
    ar.add_argument('--satellite', default='noaa20', choices=['noaa20', 'snpp'])
    b = sub.add_parser('build', help='Build the labelled dataset from an ingest directory')
    b.add_argument('--input', required=True)
    b.add_argument('--out', default=None, help='(API-mode only) CSV path')
    for name, helptext in [('experiment', 'Validation-only model comparison; freezes the champion'),
                           ('train', 'Alias of experiment')]:
        e = sub.add_parser(name, help=helptext)
        e.add_argument('--data', required=True)
        e.add_argument('--out', default='output')
        e.add_argument('--skip-regression', action='store_true')
    t = sub.add_parser('evaluate-test', help='ONE-TIME held-out test of the frozen champion')
    t.add_argument('--data', required=True)
    t.add_argument('--out', default='output')
    v = sub.add_parser('vintage-check', help='ERA5 vs archived operational forecasts; type-less NRT label proxy')
    v.add_argument('--input', default='data/norcal')
    v.add_argument('--data', required=True)
    v.add_argument('--out', default='output')
    lv = sub.add_parser('predict-live-v02', help='(v0.2, NorCal only) score with NRT + forecast weather')
    region(lv, 'data/live')
    lv.add_argument('--model', default='output/model.joblib')
    lv.add_argument('--predictions', default='output/predictions.json')
    lv.add_argument('--static-mask', default='data/norcal/interim/static_source_mask_2019_2022.json')
    # ---------------- v0.3 multi-region
    i3 = sub.add_parser('ingest-v3', help='v0.3 keyless multi-region ingest (FIRMS, POWER, WorldCover, DEM)')
    i3.add_argument('--out', default='data/v03')
    b3 = sub.add_parser('build-v3', help='v0.3 dataset release (test labels withheld unless --reveal-test-labels)')
    b3.add_argument('--root', default='data/v03')
    b3.add_argument('--reveal-test-labels', action='store_true')
    e3 = sub.add_parser('experiment-v3', help='v0.3 CV selection on development rows only; freezes champion')
    e3.add_argument('--data', default='data/v03/dataset/firerisk_open_multiregion_v0.3.0.parquet')
    e3.add_argument('--out', default='output_v3')
    t3 = sub.add_parser('evaluate-test-v3', help='v0.3 ONE-TIME untouched test')
    t3.add_argument('--data', default='data/v03/dataset/firerisk_open_multiregion_v0.3.0.parquet')
    t3.add_argument('--out', default='output_v3')
    l3 = sub.add_parser('predict-live', help='v0.3: score today\'s 00:00 UTC issue in all supported regions')
    l3.add_argument('--model', default='output_v3/model_v3.joblib')
    l3.add_argument('--predictions', default='output_v3/predictions.json')
    l3.add_argument('--live-dir', default='data/live')
    l3.add_argument('--regions', default=None, help='comma-separated subset of regions')
    args = p.parse_args(argv)
    {'ingest': ingest, 'ingest-archive': ingest_archive_cmd, 'build': build, 'experiment': experiment,
     'train': experiment, 'evaluate-test': evaluate_test, 'vintage-check': vintage_check,
     'predict-live-v02': predict_live, 'ingest-v3': ingest_v3_cmd, 'build-v3': build_v3_cmd,
     'experiment-v3': experiment_v3_cmd, 'evaluate-test-v3': evaluate_test_v3_cmd,
     'predict-live': predict_live_v3_cmd}[args.command](args)


def ingest_v3_cmd(args):
    from .ingest3 import ingest_v3
    rep = ingest_v3(Path(args.out))
    print(json.dumps({k: rep[k] for k in ['firms_day_status_counts', 'raw_files', 'network_downloads_this_run']}, indent=2))


def build_v3_cmd(args):
    from .dataset3 import build_release_v3
    rep = build_release_v3(Path(args.root), reveal_test_labels=args.reveal_test_labels)
    print(json.dumps({k: rep[k] for k in ['rows', 'in_primary_analysis', 'row_accounting']}, indent=2))


def experiment_v3_cmd(args):
    from .experiment_v3 import regression_v3, run_development_v3
    out = Path(args.out)
    if (out / 'test_results_v3.json').exists():
        raise SystemExit('Untouched test already run for this output directory: development is closed.')
    df = pd.read_parquet(args.data)
    df = df[df['in_primary_analysis']]
    run_development_v3(df, out)
    regression_v3(df, out)


def evaluate_test_v3_cmd(args):
    from .experiment_v3 import run_final_test_v3
    df = pd.read_parquet(args.data)
    res = run_final_test_v3(df[df['in_primary_analysis']], Path(args.out))
    print(json.dumps({r: res[r]['metrics']['champion']['average_precision'] for r in res
                      if isinstance(res[r], dict) and 'metrics' in res[r]}, indent=2))


def predict_live_v3_cmd(args):
    from .live_v3 import run_live_v3
    st = run_live_v3(Path(args.model), Path(args.live_dir), Path(args.predictions),
                     regions=args.regions.split(',') if args.regions else None, key=load_env_key())
    if st['state'] != 'OK':
        raise SystemExit(f"{st['state']}: {'; '.join(st['reasons'])}")


if __name__ == '__main__':
    main()
