"""v0.3 experiment runner (rules frozen in docs/PREREGISTRATION_v0.3.md).

``run_development_v3`` sees ONLY development rows (development regions, targets 2019-01-02..2023-12-31).
``run_final_test_v3`` loads the frozen artefacts and evaluates the untouched test sets exactly once.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .experiment import _cell_xy, classification_metrics, regression_metrics, reliability_table
from .features_v3 import FEATURE_SETS, LOG_FEATURES, SET_A, SET_B, TARGET
from .regions import DEV_REGIONS, PREVIOUSLY_TESTED, TEST_SPACE_REGIONS

SEED = 17
EMBARGO_DAYS = 32
DEV_FIRST_TARGET = '2019-01-02'
FOLD_YEARS = [2021, 2022, 2023]
TEST_WINDOW = ('2024-02-02', '2025-12-31')

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAVE_XGB = True
except Exception:          # libomp missing on macOS: see scripts/macos_xgboost_libomp.sh
    HAVE_XGB = False


# ----------------------------------------------------------------------------- roles
def assign_roles(df: pd.DataFrame) -> pd.Series:
    t = pd.to_datetime(df['target_date'])
    role = pd.Series('unused', index=df.index)
    dev = df['region'].isin(DEV_REGIONS)
    tsp = df['region'].isin(TEST_SPACE_REGIONS)
    first = pd.Timestamp(DEV_FIRST_TARGET)
    a, b = pd.Timestamp(TEST_WINDOW[0]), pd.Timestamp(TEST_WINDOW[1])
    role[dev & (t >= first) & (t <= pd.Timestamp('2023-12-31'))] = 'dev'
    role[dev & (t >= a) & (t <= b)] = 'test_time'
    for r, (s, e) in PREVIOUSLY_TESTED.items():
        role[(df['region'] == r) & (t >= a) & (t <= b)] = 'previously_tested_norcal'
    role[tsp & (t >= a) & (t <= b)] = 'test_space_primary'
    role[tsp & (t >= first) & (t <= pd.Timestamp('2023-12-31'))] = 'test_space_secondary'
    return role


def fold_masks(dev: pd.DataFrame, year: int):
    t = pd.to_datetime(dev['target_date'])
    tr = t <= pd.Timestamp(f'{year-1}-12-31')
    va = (t >= pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=EMBARGO_DAYS)) & (t <= pd.Timestamp(f'{year}-12-31'))
    return tr.to_numpy(), va.to_numpy()


# ----------------------------------------------------------------------------- baselines
class HistClimatology(BaseEstimator, ClassifierMixin):
    """Isotonic map from the cell's S-NPP 2012-2017 fire-days in the issue month to probability.
    Uses no study-period information about the cell, so it is valid in unseen regions."""

    def fit(self, X, y):
        self.iso_ = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(
            np.asarray(X['clim_fire_days_month'], float), np.asarray(y, float))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.iso_.predict(np.asarray(X['clim_fire_days_month'], float))
        return np.column_stack([1 - p, p])


class TrainClimatology(BaseEstimator, ClassifierMixin):
    """Training-label rate per (cell_key, month), shrunk to the month rate; month rate for unseen cells."""

    def __init__(self, prior_strength: float = 20.0):
        self.prior_strength = prior_strength

    @staticmethod
    def _month(X):
        return (np.floor((np.arctan2(X['doy_sin'], X['doy_cos']) % (2*np.pi)) / (2*np.pi) * 12).astype(int).clip(0, 11))

    def fit(self, X, y):
        d = pd.DataFrame({'c': X['cell_key'].to_numpy(), 'm': self._month(X).to_numpy(), 'y': np.asarray(y, float)})
        self.month_ = d.groupby('m')['y'].mean().to_dict()
        self.glob_ = float(d['y'].mean())
        g = d.groupby(['c', 'm'])['y'].agg(['sum', 'count'])
        mr = np.array([self.month_[m] for m in g.index.get_level_values('m')])
        self.cm_ = ((g['sum'] + self.prior_strength*mr) / (g['count'] + self.prior_strength)).to_dict()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        m = self._month(X).to_numpy()
        p = np.array([self.cm_.get((c, mm), self.month_.get(mm, self.glob_)) for c, mm in zip(X['cell_key'], m)])
        return np.column_stack([1 - p, p])


# ----------------------------------------------------------------------------- candidates
def _prep(features, scale):
    logc = [c for c in features if c in LOG_FEATURES]
    other = [c for c in features if c not in LOG_FEATURES]
    ct = ColumnTransformer([('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one'), logc),
                            ('id', 'passthrough', other)], verbose_feature_names_out=False)
    steps = [('cols', ct), ('impute', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler()))
    return steps


def _grid(**kw):
    return [dict(zip(kw, v)) for v in itertools.product(*kw.values())]


def candidates() -> dict:
    """name -> (factory, param grid, needs scaling, kind). kind: 'learned' (fit on A and B) or 'baseline'."""
    c = {
        'dummy_prior': (lambda: DummyClassifier(strategy='prior'), [{}], False, 'baseline'),
        'dummy_most_frequent': (lambda: DummyClassifier(strategy='most_frequent'), [{}], False, 'baseline'),
        'dummy_stratified': (lambda: DummyClassifier(strategy='stratified', random_state=SEED), [{}], False, 'baseline'),
        'climatology_hist_snpp': (lambda: HistClimatology(), [{}], False, 'baseline'),
        'climatology_train_cell_month': (lambda: TrainClimatology(), [{}], False, 'baseline'),
        'logistic_l2': (lambda: LogisticRegression(max_iter=3000, random_state=SEED), _grid(C=[0.01, 0.1, 1.0]), True, 'learned'),
        'gaussian_nb': (lambda: GaussianNB(), [{}], True, 'learned'),
        'decision_tree': (lambda: DecisionTreeClassifier(min_samples_leaf=100, random_state=SEED),
                          _grid(max_depth=[4, 8]), False, 'learned'),
        'random_forest': (lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1,
                                                         random_state=SEED), _grid(max_depth=[12, None]), False, 'learned'),
        'extra_trees': (lambda: ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
                        _grid(min_samples_leaf=[20, 50]), False, 'learned'),
        'hist_gradient_boosting': (lambda: HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, l2_regularization=1.0, min_samples_leaf=40, early_stopping=False,
            random_state=SEED), _grid(max_leaf_nodes=[15, 31]), False, 'learned'),
        'mlp': (lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True,
                                      random_state=SEED), _grid(alpha=[1e-3, 1e-2]), True, 'learned'),
    }
    if HAVE_XGB:
        c['xgboost'] = (lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, subsample=0.8,
                                              colsample_bytree=0.8, min_child_weight=5, n_jobs=-1,
                                              random_state=SEED, eval_metric='aucpr'),
                        _grid(max_depth=[4, 6]), False, 'learned')
    return c


BASELINE_COLS = {'climatology_hist_snpp': ['clim_fire_days_month'],
                 'climatology_train_cell_month': ['cell_key', 'doy_sin', 'doy_cos']}


def build(name: str, fs: str, params: dict) -> tuple[Pipeline, list]:
    fac, _, scale, kind = candidates()[name]
    est = fac().set_params(**params)
    if kind == 'baseline':
        cols = BASELINE_COLS.get(name, SET_A[:2])
        return Pipeline([('model', est)]), cols
    feats = FEATURE_SETS[fs]
    return Pipeline(_prep(feats, scale) + [('model', est)]), feats


# ----------------------------------------------------------------------------- metrics helpers
def week_clusters(df):
    return pd.to_datetime(df['target_date']).dt.strftime('%G-W%V').to_numpy()


def bootstrap(y, preds: dict, clusters, n=1000, seed=SEED, pairs=()):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    uc, inv = np.unique(np.asarray(clusters), return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uc))]
    ap = {k: [] for k in preds}
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uc), len(uc))])
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        for k, p in preds.items():
            ap[k].append(average_precision_score(yy, np.asarray(p)[idx]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    out = {k: {'ap_ci95': ci(v), 'ap_se': float(np.std(v))} for k, v in ap.items()}
    for a, b in pairs:
        d = np.array(ap[a]) - np.array(ap[b])
        out[f'delta_ap[{a} - {b}]'] = {'ci95': ci(d), 'share_gt_0': float((d > 0).mean())}
    out['_clusters'], out['_replicates'] = int(len(uc)), int(len(next(iter(ap.values()))))
    return out


def event_metrics(df: pd.DataFrame, p: np.ndarray, thr: float) -> dict:
    """Events = positive cell-days linked by 8-neighbour adjacency within ≤2 days (per region)."""
    d = df[['region', 'cell_id', 'target_date', TARGET]].copy()
    d['p'] = p
    d['alert'] = d['p'] >= thr
    pos = d[d[TARGET] == 1].copy()
    if pos.empty:
        return {'events': 0}
    y, x = _cell_xy(pos['cell_id'])
    t = pd.to_datetime(pos['target_date']).map(pd.Timestamp.toordinal).to_numpy()
    keys = list(zip(pos['region'], y, x, t))
    idx = {k: i for i, k in enumerate(keys)}
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, (rg, yy, xx, tt) in enumerate(keys):
        for dy, dx, dt in itertools.product((-1, 0, 1), (-1, 0, 1), (0, 1, 2)):
            j = idx.get((rg, yy+dy, xx+dx, tt+dt))
            if j is not None:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    pos['event'] = [find(i) for i in range(len(keys))]
    pos['t'] = t
    first = pos.groupby('event')['t'].transform('min')
    onset = pos[pos['t'] == first].groupby('event')['alert'].any()
    anyhit = pos.groupby('event')['alert'].any()
    size = pos.groupby('event').size()
    return {'events': int(len(onset)), 'onset_recall': float(onset.mean()), 'event_recall': float(anyhit.mean()),
            'onset_recall_multi_cellday_events': float(onset[size > 1].mean()) if (size > 1).any() else None,
            'alerts_per_1000_cell_days': float(1000 * d['alert'].mean()), 'cell_days': int(len(d))}


def _crossfit_calibrated(y, p, fold_id):
    """Isotonic calibration + F1 threshold fit on the other folds, applied to each fold (honest checks)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    cal = np.empty_like(p)
    yhat = np.zeros_like(y)
    for f in np.unique(fold_id):
        o, m = fold_id != f, fold_id == f
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p[o], y[o])
        po = iso.predict(p[o])
        qs = np.unique(np.quantile(po, np.linspace(0.5, 0.999, 200)))
        thr = max(qs, key=lambda t: f1_score(y[o], po >= t, zero_division=0))
        cal[m] = iso.predict(p[m])
        yhat[m] = cal[m] >= thr
    return cal, yhat


def _hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


# ----------------------------------------------------------------------------- development
def run_development_v3(df_all: pd.DataFrame, outdir: Path, log=print) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    roles = assign_roles(df_all)
    dev = df_all[(roles == 'dev') & df_all[TARGET].notna()].reset_index(drop=True)
    assert dev['region'].isin(DEV_REGIONS).all() and pd.to_datetime(dev['target_date']).max() <= pd.Timestamp('2023-12-31')
    y = dev[TARGET].astype(int).to_numpy()
    folds = {yr: fold_masks(dev, yr) for yr in FOLD_YEARS}
    oof_mask = np.zeros(len(dev), bool)
    fold_id = np.full(len(dev), -1)
    for yr, (_, va) in folds.items():
        oof_mask |= va
        fold_id[va] = yr
    summary = {}
    for yr, (tr, va) in folds.items():
        summary[yr] = {'train_rows': int(tr.sum()), 'train_pos': int(y[tr].sum()),
                       'val_rows': int(va.sum()), 'val_pos': int(y[va].sum())}
    log(f'dev rows {len(dev):,}; folds {summary}')

    rows, oof_store, fold2023_models = [], {}, {}
    for name, (_, grid, _, kind) in candidates().items():
        fsets = ['n/a'] if kind == 'baseline' else list(FEATURE_SETS)
        for fs, params in itertools.product(fsets, grid):
            t0 = time.time()
            oof = np.full(len(dev), np.nan)
            for yr, (tr, va) in folds.items():
                pipe, cols = build(name, fs, params)
                pipe.fit(dev.loc[tr, cols], y[tr])
                oof[va] = pipe.predict_proba(dev.loc[va, cols])[:, 1]
                if yr == 2023:
                    m2023 = pipe
            ap = average_precision_score(y[oof_mask], oof[oof_mask])
            per_fold = {yr: float(average_precision_score(y[va], oof[va])) for yr, (_, va) in folds.items()}
            for reg in DEV_REGIONS:   # reporting only (selection uses pooled AP, as pre-registered)
                mr = oof_mask & (dev['region'] == reg).to_numpy()
                per_fold[f'region_{reg}'] = float(average_precision_score(y[mr], oof[mr])) if y[mr].sum() else None
            key = (name, fs, json.dumps(params))
            rows.append({'model': name, 'feature_set': fs, 'params': json.dumps(params), 'kind': kind,
                         'oof_ap_pooled': float(ap), **{f'ap_{k}': v for k, v in per_fold.items()},
                         'seconds': round(time.time() - t0, 1)})
            oof_store[key], fold2023_models[key] = oof, m2023
            log(f"  {name:28s} {fs:20s} {json.dumps(params):32s} pooled OOF AP={ap:.4f}  ({rows[-1]['seconds']}s)")
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(outdir / 'cv_grid_all_configs.csv', index=False)
    best = grid_df.loc[grid_df.groupby(['model', 'feature_set'])['oof_ap_pooled'].idxmax()].copy()

    # honest cross-fitted checks
    yo, fo = y[oof_mask], fold_id[oof_mask]
    weeks = week_clusters(dev[oof_mask])
    checks = []
    for r in best.itertuples():
        p = oof_store[(r.model, r.feature_set, r.params)][oof_mask]
        cal, yhat = _crossfit_calibrated(yo, p, fo)
        m = classification_metrics(yo, cal, 0.5)
        m.update({'balanced_accuracy': float(((yhat == 1) & (yo == 1)).sum() / max(1, yo.sum()) / 2 +
                                             ((yhat == 0) & (yo == 0)).sum() / max(1, (yo == 0).sum()) / 2),
                  'f1': float(f1_score(yo, yhat, zero_division=0)),
                  'recall': float(((yhat == 1) & (yo == 1)).sum() / max(1, yo.sum())),
                  'precision': float(((yhat == 1) & (yo == 1)).sum() / max(1, yhat.sum())),
                  'accuracy': float((yhat == yo).mean())})
        checks.append({'model': r.model, 'feature_set': r.feature_set, 'cf_brier_calibrated': m['brier'],
                       'cf_balanced_accuracy': m['balanced_accuracy'], 'cf_f1': m['f1'], 'cf_recall': m['recall'],
                       'cf_precision': m['precision'], 'cf_accuracy': m['accuracy'],
                       'oof_roc_auc': float(roc_auc_score(yo, p)) if len(set(p)) > 1 else 0.5})
    board = best.merge(pd.DataFrame(checks), on=['model', 'feature_set']).sort_values('oof_ap_pooled', ascending=False)

    # 1-SE rule
    top = board.iloc[0]
    top_key = (top['model'], top['feature_set'], top['params'])
    se = bootstrap(yo, {'top': oof_store[top_key][oof_mask]}, weeks, n=500)['top']['ap_se']
    board['within_1se_of_best'] = board['oof_ap_pooled'] >= top['oof_ap_pooled'] - se
    # leave-one-region-out for: candidates within 1 SE + their matched partners + baselines (reporting)
    learned_close = board[board.within_1se_of_best & (board.kind == 'learned')]
    loro_keys = set()
    for r in learned_close.itertuples():
        for fs in FEATURE_SETS:
            row = board[(board.model == r.model) & (board.feature_set == fs)].iloc[0]
            loro_keys.add((row.model, row.feature_set, row.params))
    for r in board[board.kind == 'baseline'].itertuples():
        loro_keys.add((r.model, r.feature_set, r.params))
    t = pd.to_datetime(dev['target_date'])
    tr_all = (t <= pd.Timestamp('2022-12-31')).to_numpy()
    va_all = ((t >= pd.Timestamp('2023-01-01') + pd.Timedelta(days=EMBARGO_DAYS)) & (t <= pd.Timestamp('2023-12-31'))).to_numpy()
    loro = {}
    for key in sorted(loro_keys):
        name, fs, params = key
        preds, ys, per_region = [], [], {}
        for reg in DEV_REGIONS:
            trm = tr_all & (dev['region'] != reg).to_numpy()
            vam = va_all & (dev['region'] == reg).to_numpy()
            pipe, cols = build(name, fs, json.loads(params))
            pipe.fit(dev.loc[trm, cols], y[trm])
            pp = pipe.predict_proba(dev.loc[vam, cols])[:, 1]
            preds.append(pp)
            ys.append(y[vam])
            per_region[reg] = float(average_precision_score(y[vam], pp))
        loro[key] = {'loro_ap_pooled': float(average_precision_score(np.concatenate(ys), np.concatenate(preds))),
                     **{f'loro_ap_{k}': v for k, v in per_region.items()}}
        log(f"  LORO {name:28s} {fs:20s} pooled AP={loro[key]['loro_ap_pooled']:.4f} {per_region}")
    lor = pd.DataFrame([{'model': k[0], 'feature_set': k[1], **v} for k, v in loro.items()])
    board = board.merge(lor, on=['model', 'feature_set'], how='left')
    board.to_csv(outdir / 'validation_leaderboard_v3.csv', index=False)

    elig = board[board.within_1se_of_best & board['loro_ap_pooled'].notna()]
    champ = elig.sort_values(['loro_ap_pooled', 'cf_brier_calibrated'], ascending=[False, True]).iloc[0]
    ckey = (champ['model'], champ['feature_set'], champ['params'])
    # matched fire-history ablation: champion family if learned, else the best learned family by OOF AP
    fam = champ['model'] if champ['kind'] == 'learned' else board[board.kind == 'learned'].iloc[0]['model']
    pair = {}
    for fs in FEATURE_SETS:
        row = board[(board.model == fam) & (board.feature_set == fs)].iloc[0]
        pair[fs] = (row['model'], row['feature_set'], row['params'])
    ab = bootstrap(yo, {fs: oof_store[k][oof_mask] for fs, k in pair.items()}, weeks, n=1000,
                   pairs=[('B_plus_fire_history', 'A_no_fire_info')])

    # freeze: fold-2023 models (trained on targets <= 2022) + isotonic/threshold on 2023
    va23 = folds[2023][1]
    frozen = {}
    for r in board.itertuples():
        key = (r.model, r.feature_set, r.params)
        p23 = oof_store[key][va23]
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p23, y[va23])
        c23 = iso.predict(p23)
        qs = np.unique(np.quantile(c23, np.linspace(0.5, 0.999, 300)))
        thr = float(max(qs, key=lambda t_: f1_score(y[va23], c23 >= t_, zero_division=0))) if len(qs) else 0.5
        top5 = float(np.quantile(c23, 0.95))
        _, cols = build(r.model, r.feature_set, json.loads(r.params))
        frozen[f'{r.model}|{r.feature_set}'] = {'pipeline': fold2023_models[key], 'calibrator': iso, 'columns': cols,
                                               'threshold_f1': thr, 'threshold_top5pct': top5, 'params': r.params}
    joblib.dump(frozen, outdir / 'frozen_candidates_v3.joblib')
    cfz = frozen[f"{champ['model']}|{champ['feature_set']}"]
    imp_cols = cfz['columns']
    Xv, yv = dev.loc[va23, imp_cols].reset_index(drop=True), y[va23]
    base_ap = average_precision_score(yv, cfz['pipeline'].predict_proba(Xv)[:, 1])
    rng = np.random.default_rng(SEED)
    drops = {c: [] for c in imp_cols}
    for c in imp_cols:
        for _ in range(5):
            Xp = Xv.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops[c].append(base_ap - average_precision_score(yv, cfz['pipeline'].predict_proba(Xp)[:, 1]))
    pd.DataFrame({'feature': imp_cols, 'perm_importance_val2023_ap_mean': [np.mean(drops[c]) for c in imp_cols],
                  'perm_importance_val2023_ap_std': [np.std(drops[c]) for c in imp_cols]}) \
        .sort_values('perm_importance_val2023_ap_mean', ascending=False) \
        .to_csv(outdir / 'champion_permutation_importance_v3.csv', index=False)
    bounds = {r: list(v) for r, v in __import__('fireaq.regions', fromlist=['REGIONS']).REGIONS.items()}
    meta = {
        'status': 'experimental_research_model', 'version': '0.3.0',
        'frozen_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'selection_rule': 'pooled OOF AP over CV folds 2021-2023; among candidates within 1 bootstrap SE of the best, '
                          'highest leave-one-region-out AP; tie -> lower cross-fitted calibrated Brier',
        'best_oof_ap': float(top['oof_ap_pooled']), 'one_se': se,
        'champion': {'key': f"{champ['model']}|{champ['feature_set']}", 'model': champ['model'],
                     'feature_set': champ['feature_set'], 'params': json.loads(champ['params']),
                     'features': imp_cols, 'oof_ap_pooled': float(champ['oof_ap_pooled']),
                     'loro_ap_pooled': float(champ['loro_ap_pooled'])},
        'ablation_family': fam,
        'ablation_pair': {fs: f'{k[0]}|{k[1]}' for fs, k in pair.items()},
        'validation_matched_ablation_bootstrap': ab,
        'threshold_calibrated_f1_2023': cfz['threshold_f1'],
        'cv_fold_summary': summary,
        'supported_regions': {r: {'bbox': REGIONS_bbox, 'role': role}
                              for r, REGIONS_bbox, role in [(k, v['bbox'], v['role']) for k, v in
                                                            __import__('fireaq.regions', fromlist=['REGIONS']).REGIONS.items()]},
        'dev_dataset_sha256': _hash(dev),
        'test_evaluated': False,
    }
    del bounds
    joblib.dump({'model': cfz['pipeline'], 'calibrator': cfz['calibrator'], 'features': imp_cols,
                 'threshold': cfz['threshold_f1'], 'metadata': meta}, outdir / 'model_v3.joblib')
    (outdir / 'frozen_champion_v3.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')
    log(f"CHAMPION {meta['champion']['key']} OOF AP={champ['oof_ap_pooled']:.4f} LORO AP={champ['loro_ap_pooled']:.4f} (1SE={se:.4f})")
    return meta


def regression_v3(df_all: pd.DataFrame, outdir: Path, log=print, test: bool = False) -> pd.DataFrame:
    """Secondary continuous target log1p(next-day count); train <= 2022 dev, validate 2023 dev; optional one-time test."""
    outdir = Path(outdir)
    roles = assign_roles(df_all)
    dev = df_all[(roles == 'dev') & df_all['target_nextday_count'].notna()]
    t = pd.to_datetime(dev['target_date'])
    tr = dev[t <= pd.Timestamp('2022-12-31')]
    va = dev[(t >= pd.Timestamp('2023-01-01') + pd.Timedelta(days=EMBARGO_DAYS))]
    f = SET_B
    pre = lambda s: _prep(f, s)  # noqa: E731
    models = {'mean_constant': Pipeline([('model', DummyRegressor(strategy='mean'))]),
              'median_constant': Pipeline([('model', DummyRegressor(strategy='median'))]),
              'linear_regression': Pipeline(pre(True) + [('model', LinearRegression())]),
              'ridge_a10': Pipeline(pre(True) + [('model', Ridge(alpha=10.0))]),
              'random_forest_regressor': Pipeline(pre(False) + [('model', RandomForestRegressor(
                  n_estimators=150, min_samples_leaf=20, max_depth=16, n_jobs=-1, random_state=SEED))]),
              'hist_gradient_boosting_regressor': Pipeline(pre(False) + [('model', HistGradientBoostingRegressor(
                  max_iter=300, learning_rate=0.05, early_stopping=False, random_state=SEED))]),
              'mlp_regressor': Pipeline(pre(True) + [('model', MLPRegressor(
                  hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=200, random_state=SEED))])}
    if HAVE_XGB:
        models['xgboost_regressor'] = Pipeline(pre(False) + [('model', XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=SEED))])
    rows = []
    path = outdir / 'regression_frozen_v3.joblib'
    if test:
        models = joblib.load(path)
        for role in ['test_time', 'test_space_primary']:
            te = df_all[(roles == role) & df_all['target_nextday_count'].notna()]
            yt = np.log1p(te['target_nextday_count'].astype(float))
            for n, m in models.items():
                rows.append({'model': n, 'test_set': role, **{f'test_{k}': v for k, v in
                                                               regression_metrics(yt, m.predict(te[f])).items()}})
        return pd.DataFrame(rows)
    ytr, yva = np.log1p(tr['target_nextday_count'].astype(float)), np.log1p(va['target_nextday_count'].astype(float))
    for n, m in models.items():
        m.fit(tr[f], ytr)
        rows.append({'model': n, **{f'val_{k}': v for k, v in regression_metrics(yva, m.predict(va[f])).items()}})
        log(f"  regression {n:34s} val RMSE={rows[-1]['val_RMSE']:.4f} R2={rows[-1]['val_R2']:.4f}")
    joblib.dump(models, path)
    out = pd.DataFrame(rows).sort_values('val_RMSE')
    out.to_csv(outdir / 'regression_validation_v3.csv', index=False)
    return out


# ----------------------------------------------------------------------------- final test (once)
def run_final_test_v3(df_all: pd.DataFrame, outdir: Path, log=print) -> dict:
    outdir = Path(outdir)
    res_path = outdir / 'test_results_v3.json'
    if res_path.exists():
        raise RuntimeError(f'{res_path} exists: the v0.3 untouched test has already been run once.')
    bundle = joblib.load(outdir / 'model_v3.joblib')
    frozen = joblib.load(outdir / 'frozen_candidates_v3.joblib')
    meta = bundle['metadata']
    roles = assign_roles(df_all)
    dev = df_all[(roles == 'dev') & df_all[TARGET].notna()].reset_index(drop=True)
    if _hash(dev) != meta['dev_dataset_sha256']:
        raise RuntimeError('Development rows changed since freezing; refusing to evaluate.')
    ckey = meta['champion']['key']
    comps = {'champion': ckey, 'ablation_B_plus_fire_history': meta['ablation_pair']['B_plus_fire_history'],
             'ablation_A_no_fire_info': meta['ablation_pair']['A_no_fire_info'],
             'climatology_hist_snpp': 'climatology_hist_snpp|n/a',
             'climatology_train_cell_month': 'climatology_train_cell_month|n/a', 'dummy_prior': 'dummy_prior|n/a'}
    pairs = [('champion', 'climatology_hist_snpp'), ('champion', 'climatology_train_cell_month'),
             ('ablation_B_plus_fire_history', 'ablation_A_no_fire_info')]

    def score(key, frame):
        fz = frozen[key]
        raw = fz['pipeline'].predict_proba(frame[fz['columns']])[:, 1]
        return fz['calibrator'].predict(raw)

    results = {'evaluated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'champion': ckey,
               'ablation_pair': meta['ablation_pair']}
    all_preds = []
    for role in ['test_time', 'test_space_primary', 'test_space_secondary', 'previously_tested_norcal']:
        te = df_all[(roles == role) & df_all[TARGET].notna()].reset_index(drop=True)
        if te.empty or te[TARGET].sum() == 0:
            results[role] = {'note': 'no labelled rows or no positives'}
            continue
        yt = te[TARGET].astype(int).to_numpy()
        preds = {k: score(v, te) for k, v in comps.items()}
        block = {'rows': int(len(te)), 'positives': int(yt.sum()), 'prevalence': float(yt.mean()),
                 'regions': sorted(te['region'].unique().tolist()),
                 'target_dates': [te['target_date'].min(), te['target_date'].max()]}
        block['metrics'] = {k: classification_metrics(yt, p, frozen[comps[k]]['threshold_f1']) for k, p in preds.items()}
        block['bootstrap_week'] = bootstrap(yt, preds, week_clusters(te), pairs=pairs)
        block['bootstrap_cell'] = bootstrap(yt, preds, te['cell_key'].to_numpy(), pairs=pairs)
        block['events_at_f1_threshold'] = {k: event_metrics(te, p, frozen[comps[k]]['threshold_f1'])
                                           for k, p in preds.items()}
        block['events_at_matched_top5pct_threshold'] = {k: event_metrics(te, p, frozen[comps[k]]['threshold_top5pct'])
                                                        for k, p in preds.items()}
        block['by_region'] = {}
        for reg, g in te.groupby('region'):
            ix = g.index.to_numpy()
            if yt[ix].sum():
                block['by_region'][reg] = {k: {'ap': float(average_precision_score(yt[ix], p[ix])),
                                               'roc_auc': float(roc_auc_score(yt[ix], p[ix])),
                                               'n': int(len(ix)), 'positives': int(yt[ix].sum())}
                                           for k, p in preds.items()}
        block['reliability_champion'] = reliability_table(yt, preds['champion'])
        sens = {}
        for name, m in {'exclude_cloud_obscured_negatives': ~(te['target_cloud_obscured_proxy'].astype(bool) & (yt == 0)),
                        'no_issue_day_detection': ~te['detection_on_issue_day'].astype(bool)}.items():
            m = np.asarray(m)
            sens[name] = {k: float(average_precision_score(yt[m], p[m])) for k, p in preds.items()}
        block['sensitivity_ap'] = sens
        board = []
        for key, fz in frozen.items():
            p = score(key, te)
            board.append({'model': key, 'ap': float(average_precision_score(yt, p)),
                          'roc_auc': float(roc_auc_score(yt, p)) if len(set(p)) > 1 else 0.5,
                          'brier': float(brier_score_loss(yt, p))})
        block['all_frozen_candidates'] = sorted(board, key=lambda r: -r['ap'])
        results[role] = block
        pr = te[['region', 'cell_key', 'issue_date', 'target_date', TARGET, 'target_nextday_count',
                 'detection_on_issue_day', 'target_cloud_obscured_proxy']].copy()
        pr['role'] = role
        for k, p in preds.items():
            pr[f'p_{k}'] = p
        all_preds.append(pr)
        c = block['metrics']['champion']
        log(f"{role}: n={len(te):,} pos={yt.sum():,} champion AP={c['average_precision']:.4f} "
            f"histclim AP={block['metrics']['climatology_hist_snpp']['average_precision']:.4f} "
            f"B AP={block['metrics']['ablation_B_plus_fire_history']['average_precision']:.4f} "
            f"A AP={block['metrics']['ablation_A_no_fire_info']['average_precision']:.4f}")
    pd.concat(all_preds, ignore_index=True).to_parquet(outdir / 'test_predictions_v3.parquet', index=False)
    reg = regression_v3(df_all, outdir, test=True)
    results['regression_log1p_count_test'] = reg.to_dict(orient='records')
    res_path.write_text(json.dumps(results, indent=2, default=str) + '\n')
    meta['test_evaluated'] = True
    meta['test_summary'] = {r: results[r]['metrics']['champion'] for r in
                            ['test_time', 'test_space_primary', 'test_space_secondary'] if 'metrics' in results.get(r, {})}
    bundle['metadata'] = meta
    joblib.dump(bundle, outdir / 'model_v3.joblib')
    (outdir / 'frozen_champion_v3.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')
    return results
