"""Temporal holdout evaluation with matched no-fire and fire-feature ablations."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import FEATURES

BASE_FEATURES = [x for x in FEATURES if not x.startswith('neighbor_')]


def metrics(y, p, cutoff=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {'n': int(len(y)), 'positives': int(y.sum()), 'prevalence': float(y.mean()),
            'average_precision': float(average_precision_score(y, p)),
            'roc_auc': float(roc_auc_score(y, p)) if len(set(y))==2 else None,
            'brier': float(brier_score_loss(y, p)),
            'f1_at_validation_threshold': float(f1_score(y, p >= cutoff,zero_division=0))}


def _candidates():
    return {
        'logistic': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                  LogisticRegression(max_iter=1000,class_weight='balanced',random_state=17)),
        'gradient_boosting': HistGradientBoostingClassifier(max_iter=120,learning_rate=0.065,
                                             l2_regularization=2,min_samples_leaf=30,random_state=17),
    }


def _select_model(train, valid, features):
    selected = None
    for name, model in _candidates().items():
        model.fit(train[features], train['target_nextday_detection'].astype(int))
        p = model.predict_proba(valid[features])[:,1]
        ap = float(average_precision_score(valid['target_nextday_detection'].astype(int),p))
        if selected is None or ap > selected['val_ap']:
            selected={'name':name,'model':model,'val_ap':ap,'val_p':p}
    return selected


def predict_prob(bundle, frame:pd.DataFrame):
    missing=set(bundle['features'])-set(frame.columns)
    if missing: raise ValueError(f'Missing prediction features: {missing}')
    bounds=bundle['metadata']['training_region_bounds']
    if (frame['latitude'].min() < bounds['min_latitude']-1e-6 or
        frame['latitude'].max() > bounds['max_latitude']+1e-6 or
        frame['longitude'].min() < bounds['min_longitude']-1e-6 or
        frame['longitude'].max() > bounds['max_longitude']+1e-6):
        raise ValueError('Cannot extrapolate beyond training geography without separate spatial validation')
    raw=bundle['model'].predict_proba(frame[bundle['features']])[:,1]
    return np.clip(bundle['calibrator'].predict(raw),0,1)


def train_and_evaluate(dataset:pd.DataFrame, outpath:Path, valid_year:int, test_year:int):
    if test_year != valid_year + 1:
        raise ValueError('First version requires consecutive validation and test years')
    df=dataset.copy()
    for col in FEATURES:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    df=df.dropna(subset=FEATURES+['target_nextday_detection'])
    df['issue_date']=pd.to_datetime(df['issue_date'])
    # Target day must be inside same split for 24-48 hour horizon. No split-boundary spillover.
    df['target_date']=pd.to_datetime(df['target_date'])
    train=df[df['target_date'] < pd.Timestamp(f'{valid_year}-01-01')]
    valid=df[(df['issue_date']>=pd.Timestamp(f'{valid_year}-01-01')) &
             (df['target_date']<pd.Timestamp(f'{test_year}-01-01'))]
    test=df[(df['issue_date']>=pd.Timestamp(f'{test_year}-01-01')) &
            (df['target_date']<pd.Timestamp(f'{test_year+1}-01-01'))]
    for name, fold in [('training',train),('validation',valid),('test',test)]:
        npos = int(fold['target_nextday_detection'].sum())
        nneg = len(fold)-npos
        if npos < (12 if name=='training' else 5) or nneg < 40:
            raise ValueError(f'{name} has {len(fold)} rows, {npos} positive labels, {nneg} negatives; '
                             'need more genuine observations before training or making claims')
    # Select a single model family on BASELINE validation results, then compare
    # matched architectures/hyperparameters on identical rows.
    base=_select_model(train,valid,BASE_FEATURES)
    fire_model=_candidates()[base['name']]
    fire_model.fit(train[FEATURES],train['target_nextday_detection'].astype(int))
    fire_val_p=fire_model.predict_proba(valid[FEATURES])[:,1]
    fire={'name':base['name'],'model':fire_model,
          'val_ap':float(average_precision_score(valid['target_nextday_detection'].astype(int),fire_val_p)),
          'val_p':fire_val_p}
    yv=valid['target_nextday_detection'].astype(int).to_numpy()
    yt=test['target_nextday_detection'].astype(int).to_numpy()
    pb_raw=base['model'].predict_proba(test[BASE_FEATURES])[:,1]
    pval=fire['val_p']
    # Explicit calibration on validation only. Keep test unseen until frozen.
    cal=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1)
    cal.fit(pval,yv)
    cal_base=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1)
    cal_base.fit(base['val_p'],yv)
    pb=np.clip(cal_base.predict(pb_raw),0,1)
    pf=np.clip(cal.predict(fire['model'].predict_proba(test[FEATURES])[:,1]),0,1)
    cutoffs=np.linspace(0.01,0.95,95)
    pc=np.clip(cal.predict(pval),0,1)
    threshold=float(max(cutoffs,key=lambda t: f1_score(yv,pc>=t,zero_division=0)))
    stats={'status':'experimental_research_model',
           'target':'qualifying VIIRS detection during UTC day d+1, among cells without known detections over prior 7 issue days',
           'issue_time':'00:00 UTC beginning issue_date',
           'prediction_horizon':'24 to 48 hours after issue midnight',
           'sources':'historical standard-product VIIRS; retrospective ERA5 via Open-Meteo',
           'limitations':['fire detections are not validated ignitions or all wildfires',
                          'non-detections include clouds, orbital gaps, and undetected fires',
                          'ERA5 historical inputs differ from live historical forecasts',
                          'no independent spatial-block or operational evaluation yet'],
           'split':{'train_target_before':f'{valid_year}-01-01',
                    'validation_issue_year':valid_year,'test_issue_year':test_year},
           'architecture_selection':'family chosen by baseline validation AP; same family/hyperparameters in both arms',
           'baseline':{'model':base['name'],'validation_ap':base['val_ap'],
                       'test':metrics(yt,pb)},
           'fire_model':{'model':fire['name'],'validation_ap':fire['val_ap'],
                         'test':metrics(yt,pf,threshold)},
           'delta_ap_test':float(average_precision_score(yt,pf)-average_precision_score(yt,pb)),
           'threshold_selected_on_validation':threshold,
           'test_evaluated_once':True,
           'training_region_bounds': {
             'min_latitude':float(df['latitude'].min()),'max_latitude':float(df['latitude'].max()),
             'min_longitude':float(df['longitude'].min()),'max_longitude':float(df['longitude'].max())},
           'train_n':len(train),'validation_n':len(valid),'test_n':len(test)}
    outpath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model':fire['model'],'calibrator':cal,'features':FEATURES,
                 'metadata':stats}, outpath)
    (outpath.parent / f'{outpath.stem}_metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
    return stats


def score_predictions(bundle_path:Path, frame:pd.DataFrame):
    obj=joblib.load(bundle_path)  # load ONLY your own trusted model files
    p=predict_prob(obj,frame)
    out=frame[['cell_id','issue_date','target_date','latitude','longitude']].copy()
    out['probability_nextday_detection']=np.round(p,6)
    out['model_status']='experimental; not an official fire forecast'
    return out,obj['metadata']
