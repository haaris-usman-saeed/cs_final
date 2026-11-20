"""FireRisk Open command line. python -m fireaq --help"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .core import Grid, make_dataset
from .model import score_predictions, train_and_evaluate
from .sources import fetch_firms,fetch_weather_historical,fetch_weather_live,write_manifest


def parse_bbox(text):
    try: coords=tuple(map(float,text.split(',')))
    except ValueError as e: raise argparse.ArgumentTypeError('Use --bbox=west,south,east,north') from e
    if len(coords)!=4: raise argparse.ArgumentTypeError('bbox needs 4 numbers')
    return coords


def get_grid(bbox, step, max_cells):
    grid=Grid(*bbox,step)
    if len(grid.cells())>max_cells:
        raise ValueError(f'Grid has {len(grid.cells())} cells; cap is {max_cells}. '
                         'Start with a smaller region, or explicitly raise --max-cells.')
    return grid


def save_csv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)


def ingest(args,live=False):
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    grid=get_grid(args.bbox,args.step,args.max_cells)
    key=os.environ.get('FIRMS_MAP_KEY','')
    if not key: raise ValueError('Set FIRMS_MAP_KEY from the NASA FIRMS site first (do not paste it into chat).')
    if live:
        issue=datetime.now(timezone.utc).date()
        # Snapshot time always 00:00 UTC of current UTC date.
        start=issue-timedelta(days=11)
        end=issue
        sources=['VIIRS_NOAA20_NRT']
    else:
        start,end=date.fromisoformat(args.start),date.fromisoformat(args.end)
        if end<start: raise ValueError('end must be >= start')
        if end>date.today()-timedelta(days=6):
            raise ValueError('For historical ERA5, end should be at least 6 days old; live mode is separate.')
        sources=args.sources.split(',')
        if any(not s.endswith('_SP') for s in sources):
            raise ValueError('Historical training requires _SP standard products; NRT is for live mode')
    manifest=[]
    # For historical studies, include all target days and feature padding.
    fire_start=start-timedelta(days=11) if not live else start
    fire_end=end+timedelta(days=1) if not live else end
    fires=[]
    for src in sources:
        df=fetch_firms(key,args.bbox,fire_start,fire_end,src,out/'raw',manifest)
        fires.append(df)
    all_fires=pd.concat(fires,ignore_index=True)
    if all_fires.empty:
        raise ValueError('FIRMS returned no rows. No zero-filled dataset will be created.')
    save_csv(all_fires,out/'firms.csv')
    weather=[]
    for row in grid.cells().itertuples():
        if live:
            wx=fetch_weather_live(row.cell_id,row.latitude,row.longitude,out/'raw',manifest)
        else:
            wx=fetch_weather_historical(row.cell_id,row.latitude,row.longitude,
                    start-timedelta(days=11),end,out/'raw',manifest)
        weather.append(wx)
        time.sleep(0.6) # conservative API pacing; underlying endpoints have fair-use limits
    save_csv(pd.concat(weather,ignore_index=True),out/'weather.csv')
    meta={
        'status':'real_downloads_not_yet_independently_verified',
        'bbox':list(args.bbox),'step':args.step, 'grid_cells':len(grid.cells()),
        'fire_source_products':sources,'fire_start':str(fire_start),'fire_end':str(fire_end),
        'issue_start':str(start),'issue_end':str(end),
        'weather_source':'Open-Meteo live forecast/past' if live else 'Open-Meteo ERA5 final reanalysis',
        'weather_data_warning':'ERA5 final reanalysis is NOT an as-issued operational forecast',
        'download_complete':True,'live':live,
        'raw_manifest_file':'manifest.json',
    }
    (out/'ingest_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    write_manifest(out/'manifest.json',manifest)
    print(f'Collected {len(all_fires):,} FIRMS rows; {len(grid.cells())} weather locations; files in {out}')
    return out


def build(args):
    root=Path(args.input)
    meta=json.loads((root/'ingest_metadata.json').read_text())
    if not meta.get('download_complete'):
        raise ValueError('Cannot build: incomplete data extraction')
    if meta['live']:
        raise ValueError('For live predictions use predict-live, not historical build')
    if date.fromisoformat(meta['fire_end'])<date.fromisoformat(meta['issue_end'])+timedelta(days=1):
        raise ValueError('FIRMS source lacks the next-day label period')
    grid=Grid(*meta['bbox'],meta['step'])
    fires=pd.read_csv(root/'firms.csv')
    wx=pd.read_csv(root/'weather.csv')
    data=make_dataset(fires,wx,grid,meta['issue_start'],meta['issue_end'])
    if data.empty: raise ValueError('No eligible training rows: inspect completeness and study coverage')
    save_csv(data,Path(args.out))
    npos=int(data['target_nextday_detection'].sum())
    print(f'Built {len(data):,} eligible cell-days, {npos:,} positive detection labels; {args.out}')


def train(args):
    df=pd.read_csv(args.data)
    summary=train_and_evaluate(df,Path(args.model),args.valid_year,args.test_year)
    print(json.dumps({'model':args.model,'train_n':summary['train_n'],
        'validation_n':summary['validation_n'],'test_n':summary['test_n'],
        'baseline_test_ap':summary['baseline']['test']['average_precision'],
        'with_fire_test_ap':summary['fire_model']['test']['average_precision'],
        'delta_ap':summary['delta_ap_test'], 'status':summary['status']},indent=2))


def predict_live(args):
    from .core import Grid
    import joblib
    model=Path(args.model)
    if not model.exists():raise ValueError('No trained model. Do not publish predictions from fabricated data.')
    import joblib
    bounds=joblib.load(model)['metadata']['training_region_bounds']
    grid=get_grid(args.bbox,args.step,args.max_cells)
    for coord in grid.cells().itertuples():
        if not (bounds['min_latitude']-1e-6 <= coord.latitude <= bounds['max_latitude']+1e-6
                and bounds['min_longitude']-1e-6 <= coord.longitude <= bounds['max_longitude']+1e-6):
            raise ValueError('Live region is outside the model training region; retrain and validate first')
    now_utc=datetime.now(timezone.utc).date()
    out=ingest(args,live=True)
    meta=json.loads((out/'ingest_metadata.json').read_text())
    grid=Grid(*meta['bbox'],meta['step'])
    raw=pd.read_csv(out/'firms.csv')
    wx=pd.read_csv(out/'weather.csv')
    dataset=make_dataset(raw,wx,grid,str(now_utc),str(now_utc),predict_only=True)
    if dataset.empty:
        raise ValueError('No forecastable cells after eligibility and feature completeness rules')
    scores,model_meta=score_predictions(model,dataset)
    obj={
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'issue_time_utc':f'{now_utc}T00:00:00Z',
        'target_date':str(now_utc+timedelta(days=1)),
        'risk_definition':'probability of qualifying VIIRS detection, NOT probability of an actual wildfire ignition',
        'forecast_status':'EXPERIMENTAL - NOT VERIFIED FOR EMERGENCY USE',
        'data_vintages':{'fires':'NASA FIRMS NRT','weather':'Open-Meteo forecast/past, differs from training ERA5'},
        'test_average_precision':model_meta['fire_model']['test']['average_precision'],
        'test_base_rate':model_meta['fire_model']['test']['prevalence'],
        'predictions':scores.to_dict(orient='records'),
    }
    path=Path(args.predictions)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n')
    print(f'Wrote {len(scores)} experimental cell probabilities to {path}')


def main():
    p=argparse.ArgumentParser(description='FireRisk Open - original data + research-grade satellite-detection ML')
    sub=p.add_subparsers(dest='command',required=True)
    def region(sp):
        sp.add_argument('--bbox',type=parse_bbox,default=(-124,40,-122,42),
                        help='west,south,east,north (e.g. --bbox=-124,40,-122,42)')
        sp.add_argument('--step',type=float,default=0.5,help='grid degrees, default 0.5')
        sp.add_argument('--max-cells',type=int,default=25)
        sp.add_argument('--out',default='data/pilot')
    a=sub.add_parser('ingest',help='Download genuine historical NASA FIRMS and ERA5 data')
    region(a)
    a.add_argument('--start',required=True,help='first UTC issue day, YYYY-MM-DD')
    a.add_argument('--end',required=True,help='last UTC issue day, YYYY-MM-DD')
    a.add_argument('--sources',default='VIIRS_NOAA20_SP',help='comma separated NASA standard-product IDs')
    b=sub.add_parser('build',help='Build labelled cell/day training CSV')
    b.add_argument('--input',required=True,help='directory produced by ingest')
    b.add_argument('--out',default='data/dataset.csv')
    t=sub.add_parser('train',help='Temporal holdout train and matched no-fire ablation')
    t.add_argument('--data',required=True)
    t.add_argument('--model',default='output/model.joblib')
    t.add_argument('--valid-year',type=int,required=True)
    t.add_argument('--test-year',type=int,required=True)
    l=sub.add_parser('predict-live',help='Collect NASA NRT and weather; score at current UTC midnight')
    region(l)
    l.add_argument('--model',default='output/model.joblib')
    l.add_argument('--predictions',default='output/predictions.json')
    args=p.parse_args()
    if args.command=='ingest':ingest(args)
    elif args.command=='build':build(args)
    elif args.command=='train':train(args)
    elif args.command=='predict-live':predict_live(args)

if __name__=='__main__':main()
