"""Common, reproducible experiment runner.

Two separate stages, so the test set cannot leak into development:

1. ``run_development``: trains every candidate (bounded grid, fixed seeds) on the training split,
   scores the validation split, writes the validation leaderboard, selects the champion by the
   pre-registered criterion (validation average precision), fits isotonic calibration and the
   decision threshold on validation only, and writes ``frozen_champion.json`` + model files.
   It never reads test labels.
2. ``run_final_test``: loads the frozen artifacts, refuses to run twice, and evaluates the test
   split exactly once (all frozen candidates + baselines, bootstrap CIs, spatial blocks, error analysis).
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
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .core import COUNT_FEATURES, FEATURES, WEATHER_FEATURES

SEED = 17
TARGET = 'target_nextday_detection'
FEATURE_SETS = {'weather_only': WEATHER_FEATURES, 'weather_plus_fire': FEATURES}
SPLIT = {'train_target_end': '2022-12-31',
         'validation_targets': ('2023-01-01', '2023-12-31'),
         'test_targets': ('2024-01-01', '2025-12-31'),
         'embargo_days': 32}
SPATIAL_BLOCKS = {'lat_split': 40.5, 'lon_split': -121.5}


# ----------------------------------------------------------------------------- splits
def assign_splits(df: pd.DataFrame, split: dict = SPLIT) -> pd.Series:
    t = pd.to_datetime(df['target_date'])
    emb = pd.Timedelta(days=split['embargo_days'])
    out = pd.Series('unused', index=df.index)
    out[t <= pd.Timestamp(split['train_target_end'])] = 'train'
    for name, (a, b) in [('validation', split['validation_targets']), ('test', split['test_targets'])]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        out[(t >= a) & (t < a + emb)] = 'embargo'
        out[(t >= a + emb) & (t <= b)] = name
    return out


def spatial_block(df: pd.DataFrame) -> pd.Series:
    ns = np.where(df['latitude'] >= SPATIAL_BLOCKS['lat_split'], 'N', 'S')
    ew = np.where(df['longitude'] >= SPATIAL_BLOCKS['lon_split'], 'E', 'W')
    return pd.Series(np.char.add(ns.astype(str), ew.astype(str)), index=df.index)


def _cell_xy(cell_ids: pd.Series):
    p = cell_ids.str.extract(r'r(\d+)_c(\d+)').astype(int)
    return p[0].to_numpy(), p[1].to_numpy()


def count_events(df: pd.DataFrame) -> int:
    """Distinct positive 'events': positive cell-days linked if spatially adjacent (8-neighbour)
    and within 2 target days of each other (union-find)."""
    pos = df[df[TARGET] == 1]
    if pos.empty:
        return 0
    y, x = _cell_xy(pos['cell_id'])
    d = pd.to_datetime(pos['target_date']).map(pd.Timestamp.toordinal).to_numpy()
    keys = list(zip(y, x, d))
    idx = {k: i for i, k in enumerate(keys)}
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, (yy, xx, dd) in enumerate(keys):
        for dy, dx, dt in itertools.product((-1, 0, 1), (-1, 0, 1), (0, 1, 2)):
            j = idx.get((yy+dy, xx+dx, dd+dt))
            if j is not None:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    return len({find(i) for i in range(len(keys))})


def split_summary(df: pd.DataFrame, split_col: str = 'split') -> dict:
    out = {}
    for name, g in df.groupby(split_col):
        lab = g[TARGET].dropna()
        out[name] = {
            'rows': int(len(g)), 'labelled_rows': int(len(lab)),
            'positives': int((lab == 1).sum()), 'negatives': int((lab == 0).sum()),
            'prevalence': float(lab.mean()) if len(lab) else None,
            'negatives_per_positive': float((lab == 0).sum() / max(1, (lab == 1).sum())),
            'distinct_positive_events': count_events(g[g[TARGET].notna()]),
            'cells': int(g['cell_id'].nunique()),
            'target_date_min': g['target_date'].min(), 'target_date_max': g['target_date'].max(),
            'target_years': sorted(pd.to_datetime(g['target_date']).dt.year.unique().tolist()),
            'lat_range': [float(g['latitude'].min()), float(g['latitude'].max())],
            'lon_range': [float(g['longitude'].min()), float(g['longitude'].max())],
        }
    return out


def write_split_manifest(df: pd.DataFrame, out: Path) -> dict:
    """Explicit row-key manifest per split, with a disjointness proof."""
    out.mkdir(parents=True, exist_ok=True)
    key = df['cell_id'] + '|' + df['issue_date']
    sets = {}
    for name in ['train', 'validation', 'test', 'embargo', 'unused']:
        k = key[df['split'] == name].sort_values()
        (out / f'split_{name}_keys.txt').write_text('\n'.join(k) + ('\n' if len(k) else ''))
        sets[name] = set(k)
    if key.duplicated().any():
        raise AssertionError('duplicate cell_id|issue_date keys in dataset')
    proof = {f'{a}&{b}': len(sets[a] & sets[b]) for a, b in itertools.combinations(sets, 2)}
    if any(proof.values()):
        raise AssertionError(f'splits are not disjoint: {proof}')
    tr, va, te = (pd.to_datetime(df.loc[df.split == s, 'target_date']) for s in ['train', 'validation', 'test'])
    order = {'max_train_target': str(tr.max().date()), 'min_validation_target': str(va.min().date()),
             'max_validation_target': str(va.max().date()), 'min_test_target': str(te.min().date())}
    if not (tr.max() < va.min() and va.max() < te.min()):
        raise AssertionError(f'chronological order violated: {order}')
    manifest = {'split_definition': SPLIT, 'pairwise_key_overlap': proof, 'chronology': order,
                'sha256': {n: hashlib.sha256((out / f'split_{n}_keys.txt').read_bytes()).hexdigest()
                           for n in sets},
                'summary': split_summary(df)}
    (out / 'split_manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    for name in ['train', 'validation', 'test']:
        s = manifest['summary'].get(name)
        if not s or s['positives'] < 20 or s['negatives'] < 100:
            raise ValueError(f'split {name} lacks labels: {s}')
    return manifest


# ----------------------------------------------------------------------------- models
class Climatology(BaseEstimator, ClassifierMixin):
    """Seasonal/geographic climatology: training rate per (cell, calendar month), shrunk toward
    the region-wide monthly rate. Unseen cells fall back to the monthly rate."""

    def __init__(self, prior_strength: float = 20.0):
        self.prior_strength = prior_strength

    @staticmethod
    def _keys(X):
        month = (np.floor((np.arctan2(X['doy_sin'], X['doy_cos']) % (2*np.pi)) / (2*np.pi) * 12)
                 .astype(int).clip(0, 11))
        cell = X['latitude'].round(4).astype(str) + '_' + X['longitude'].round(4).astype(str)
        return cell.to_numpy(), month.to_numpy()

    def fit(self, X, y):
        cell, month = self._keys(X)
        d = pd.DataFrame({'cell': cell, 'month': month, 'y': np.asarray(y, float)})
        self.month_rate_ = d.groupby('month')['y'].mean().to_dict()
        self.global_rate_ = float(d['y'].mean())
        g = d.groupby(['cell', 'month'])['y'].agg(['sum', 'count'])
        m = np.array([self.month_rate_[k] for k in g.index.get_level_values('month')])
        self.cell_month_ = ((g['sum'] + self.prior_strength*m) / (g['count'] + self.prior_strength)).to_dict()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        cell, month = self._keys(X)
        p = np.array([self.cell_month_.get((c, m), self.month_rate_.get(m, self.global_rate_))
                      for c, m in zip(cell, month)])
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _prep(features, scale: bool):
    logcols = [c for c in features if c in COUNT_FEATURES]
    other = [c for c in features if c not in COUNT_FEATURES]
    ct = ColumnTransformer([('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one'), logcols),
                            ('id', 'passthrough', other)], verbose_feature_names_out=False)
    steps = [('log', ct), ('impute', SimpleImputer(strategy='median'))]   # fit on training rows only
    if scale:
        steps.append(('scale', StandardScaler()))
    return steps


def candidate_models() -> dict:
    """Frozen candidate pool: name -> (estimator factory(features) , list of param dicts, scale?)."""
    grid = lambda **kw: [dict(zip(kw, v)) for v in itertools.product(*kw.values())]  # noqa: E731
    return {
        'dummy_most_frequent': (lambda: DummyClassifier(strategy='most_frequent'), [{}], False),
        'dummy_prior': (lambda: DummyClassifier(strategy='prior'), [{}], False),
        'dummy_stratified': (lambda: DummyClassifier(strategy='stratified', random_state=SEED), [{}], False),
        'climatology_cell_month': (lambda: Climatology(), grid(prior_strength=[5.0, 20.0, 80.0]), False),
        'logistic_l2': (lambda: LogisticRegression(max_iter=3000, random_state=SEED),
                        grid(C=[0.01, 0.1, 1.0], class_weight=[None, 'balanced']), True),
        'gaussian_nb': (lambda: GaussianNB(), [{}], True),
        'decision_tree': (lambda: DecisionTreeClassifier(random_state=SEED),
                          grid(max_depth=[4, 6, 8], min_samples_leaf=[50, 200]), False),
        'random_forest': (lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
                          grid(max_depth=[8, 16, None], min_samples_leaf=[5, 20]), False),
        'extra_trees': (lambda: ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
                        grid(max_depth=[8, 16, None], min_samples_leaf=[5, 20]), False),
        'hist_gradient_boosting': (lambda: HistGradientBoostingClassifier(
                                       max_iter=250, early_stopping=False, l2_regularization=1.0,
                                       min_samples_leaf=40, random_state=SEED),
                                   grid(learning_rate=[0.03, 0.1], max_leaf_nodes=[15, 31]), False),
        'mlp_small': (lambda: MLPClassifier(max_iter=300, early_stopping=True, random_state=SEED),
                      grid(hidden_layer_sizes=[(32,), (64, 32)], alpha=[1e-4, 1e-2]), True),
    }


def make_pipeline(name: str, features: list, params: dict) -> Pipeline:
    factory, _, scale = candidate_models()[name]
    est = factory().set_params(**params)
    if name.startswith('climatology'):
        return Pipeline([('model', est)])          # needs raw lat/lon/doy columns
    return Pipeline(_prep(features, scale) + [('model', est)])


# ----------------------------------------------------------------------------- metrics
def classification_metrics(y, p, threshold: float) -> dict:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    two = len(np.unique(y)) == 2
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        'n': int(len(y)), 'positives': int(y.sum()), 'prevalence': float(y.mean()) if len(y) else None,
        'average_precision': float(average_precision_score(y, p)) if y.sum() > 0 else None,
        'roc_auc': float(roc_auc_score(y, p)) if two else None,
        'brier': float(brier_score_loss(y, p, labels=[0, 1])),
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, yhat)),
        'balanced_accuracy': float(balanced_accuracy_score(y, yhat)) if two else None,
        'precision': float(precision_score(y, yhat, zero_division=0)),
        'recall': float(recall_score(y, yhat, zero_division=0)) if y.sum() > 0 else None,
        'f1': float(f1_score(y, yhat, zero_division=0)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'single_class_split': not two,
    }


def best_f1_threshold(y, p) -> float:
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 300)))
    cands = np.unique(np.concatenate([qs, [0.5]]))
    return float(max(cands, key=lambda t: (f1_score(y, p >= t, zero_division=0), -t)))


def reliability_table(y, p, bins: int = 10) -> list:
    d = pd.DataFrame({'y': np.asarray(y, float), 'p': np.asarray(p, float)})
    d['bin'] = pd.qcut(d['p'].rank(method='first'), bins, labels=False)
    g = d.groupby('bin').agg(n=('y', 'size'), mean_predicted=('p', 'mean'), observed_rate=('y', 'mean'),
                             p_min=('p', 'min'), p_max=('p', 'max'))
    return g.reset_index().to_dict(orient='records')


def crossfit_isotonic_brier(y, p, dates) -> float:
    """Honest validation Brier: isotonic fit on one half of the validation period, scored on the other."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    first = pd.to_datetime(dates).to_numpy() < np.quantile(pd.to_datetime(dates).astype('int64'), 0.5).astype('datetime64[ns]')
    out = np.empty_like(p)
    for a, b in [(first, ~first), (~first, first)]:
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p[a], y[a])
        out[b] = iso.predict(p[b])
    return float(brier_score_loss(y, out))


# ----------------------------------------------------------------------------- development stage
def _hash_frame(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


def run_development(df: pd.DataFrame, outdir: Path, log=print) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = df[df[TARGET].notna()].copy()
    df['split'] = assign_splits(df)
    manifest = write_split_manifest(df, outdir / 'splits')
    tr, va = df[df.split == 'train'], df[df.split == 'validation']
    ytr, yva = tr[TARGET].astype(int).to_numpy(), va[TARGET].astype(int).to_numpy()
    log(f"train {len(tr):,} rows ({ytr.sum():,} pos) | validation {len(va):,} rows ({yva.sum():,} pos)")

    rows, fitted = [], {}
    for name, (_, params_list, _) in candidate_models().items():
        for fs_name, feats in FEATURE_SETS.items():
            if (name.startswith('dummy') or name.startswith('climatology')) and fs_name != 'weather_only':
                continue    # these ignore features; evaluate them once
            for params in params_list:
                t0 = time.time()
                pipe = make_pipeline(name, feats, params)
                pipe.fit(tr[feats], ytr)
                p = pipe.predict_proba(va[feats])[:, 1]
                rows.append({'model': name, 'feature_set': fs_name if not name.startswith(('dummy', 'clim'))
                             else 'n/a (baseline)', 'params': json.dumps(params, default=str),
                             'val_average_precision': float(average_precision_score(yva, p)),
                             'val_roc_auc': float(roc_auc_score(yva, p)) if len(set(p)) > 1 else 0.5,
                             'val_brier_raw': float(brier_score_loss(yva, p)),
                             'fit_seconds': round(time.time() - t0, 1)})
                fitted[(name, rows[-1]['feature_set'], rows[-1]['params'])] = (pipe, p)
                log(f"  {name:24s} {fs_name:18s} {rows[-1]['params']:55s} AP={rows[-1]['val_average_precision']:.4f}")
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(outdir / 'validation_grid_all_configs.csv', index=False)

    # best config per (model, feature set) by validation AP, then full validation metrics
    best = grid_df.loc[grid_df.groupby(['model', 'feature_set'])['val_average_precision'].idxmax()]
    board, frozen_models = [], {}
    for r in best.itertuples():
        pipe, p = fitted[(r.model, r.feature_set, r.params)]
        thr = best_f1_threshold(yva, p) if not r.model.startswith('dummy') else 0.5
        m = classification_metrics(yva, p, thr)
        m['brier_crossfit_isotonic'] = crossfit_isotonic_brier(yva, p, va['target_date'])
        board.append({'model': r.model, 'feature_set': r.feature_set, 'params': r.params,
                      **{f'val_{k}': v for k, v in m.items() if k != 'confusion_matrix'},
                      'val_confusion_matrix': json.dumps(m['confusion_matrix'])})
        frozen_models[f'{r.model}|{r.feature_set}'] = {'pipeline': pipe, 'threshold_raw': thr,
                                                       'params': r.params}
    board = pd.DataFrame(board).sort_values('val_average_precision', ascending=False)
    board.to_csv(outdir / 'validation_leaderboard.csv', index=False)

    # ---- pre-registered champion: highest validation AP (non-dummy, non-climatology preferred only by AP)
    champ = board.iloc[0]
    ckey = f"{champ['model']}|{champ['feature_set']}"
    cpipe, cval = fitted[(champ['model'], champ['feature_set'], champ['params'])]
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(cval, yva)
    cal_val = iso.predict(cval)
    threshold = best_f1_threshold(yva, cal_val)
    feats = FEATURE_SETS.get(champ['feature_set'], WEATHER_FEATURES)
    # matched ablation partner: same family + params selected for the other feature set
    partner = None
    if champ['feature_set'] in FEATURE_SETS:
        other = 'weather_only' if champ['feature_set'] == 'weather_plus_fire' else 'weather_plus_fire'
        partner = f"{champ['model']}|{other}"
    perm = permutation_importance(cpipe, va[feats], yva, scoring='average_precision', n_repeats=5,
                                  random_state=SEED, n_jobs=-1)
    importance = (pd.DataFrame({'feature': feats, 'perm_importance_val_ap_mean': perm.importances_mean,
                                'perm_importance_val_ap_std': perm.importances_std})
                  .sort_values('perm_importance_val_ap_mean', ascending=False))
    importance.to_csv(outdir / 'champion_permutation_importance_validation.csv', index=False)

    bounds = {'min_latitude': float(df['latitude'].min()), 'max_latitude': float(df['latitude'].max()),
              'min_longitude': float(df['longitude'].min()), 'max_longitude': float(df['longitude'].max())}
    meta = {
        'status': 'experimental_research_model',
        'frozen_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'selection_rule': 'highest validation average precision over all candidates and feature sets '
                          '(pre-registered in docs/PREREGISTRATION.md)',
        'champion': {'key': ckey, 'model': champ['model'], 'feature_set': champ['feature_set'],
                     'params': json.loads(champ['params']), 'features': feats,
                     'validation_average_precision': float(champ['val_average_precision'])},
        'matched_ablation_partner': partner,
        'calibration': 'isotonic regression fit on validation predictions only',
        'decision_threshold_calibrated': threshold,
        'validation_metrics_calibrated_in_sample': classification_metrics(yva, cal_val, threshold),
        'validation_reliability_calibrated_in_sample': reliability_table(yva, cal_val),
        'training_region_bounds': bounds,
        'target': 'qualifying NOAA-20 VIIRS detection in the cell during UTC day d+1, among cells without '
                  'a detection known at issue time in the previous 7 issue days',
        'issue_time': '00:00 UTC on issue_date', 'prediction_horizon': '24 to 48 hours after issue',
        'split': SPLIT, 'split_manifest_sha256': manifest['sha256'],
        'dataset_sha256_labelled_rows': _hash_frame(df.drop(columns=['split'])),
        'test_evaluated': False,
    }
    joblib.dump({'model': cpipe, 'calibrator': iso, 'features': feats, 'threshold': threshold,
                 'metadata': meta}, outdir / 'model.joblib')
    joblib.dump(frozen_models, outdir / 'frozen_candidates.joblib')
    (outdir / 'frozen_champion.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')
    log(f"CHAMPION (validation AP): {ckey} AP={champ['val_average_precision']:.4f}; frozen.")
    return meta


# ----------------------------------------------------------------------------- regression side task
def regression_candidates():
    pre = lambda scale: _prep(FEATURES, scale)  # noqa: E731
    return {
        'mean_constant': lambda: Pipeline([('model', DummyRegressor(strategy='mean'))]),
        'median_constant': lambda: Pipeline([('model', DummyRegressor(strategy='median'))]),
        'linear_regression': lambda: Pipeline(pre(True) + [('model', LinearRegression())]),
        'ridge_a1': lambda: Pipeline(pre(True) + [('model', Ridge(alpha=1.0))]),
        'ridge_a100': lambda: Pipeline(pre(True) + [('model', Ridge(alpha=100.0))]),
        'random_forest_regressor': lambda: Pipeline(pre(False) + [('model', RandomForestRegressor(
            n_estimators=200, min_samples_leaf=20, max_depth=16, n_jobs=-1, random_state=SEED))]),
        'hist_gradient_boosting_regressor': lambda: Pipeline(pre(False) + [('model', HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, early_stopping=False, random_state=SEED))]),
        'mlp_regressor': lambda: Pipeline(pre(True) + [('model', MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=300, random_state=SEED))]),
    }


def regression_metrics(y, p) -> dict:
    mse = float(mean_squared_error(y, p))
    return {'n': int(len(y)), 'MAE': float(mean_absolute_error(y, p)), 'MSE': mse,
            'RMSE': float(np.sqrt(mse)), 'R2': float(r2_score(y, p))}


def run_regression_development(df: pd.DataFrame, outdir: Path, log=print) -> pd.DataFrame:
    """Secondary continuous target: log1p(next-day qualifying detection count) on the same rows/splits."""
    df = df[df['target_nextday_count'].notna()].copy()
    df['split'] = assign_splits(df)
    tr, va = df[df.split == 'train'], df[df.split == 'validation']
    ytr, yva = np.log1p(tr['target_nextday_count'].astype(float)), np.log1p(va['target_nextday_count'].astype(float))
    rows, models = [], {}
    for name, fac in regression_candidates().items():
        m = fac().fit(tr[FEATURES], ytr)
        rows.append({'model': name, **{f'val_{k}': v for k, v in regression_metrics(yva, m.predict(va[FEATURES])).items()}})
        models[name] = m
        log(f"  regression {name:34s} val RMSE={rows[-1]['val_RMSE']:.4f} R2={rows[-1]['val_R2']:.4f}")
    board = pd.DataFrame(rows).sort_values('val_RMSE')
    board.to_csv(outdir / 'regression_validation_leaderboard.csv', index=False)
    joblib.dump(models, outdir / 'regression_frozen.joblib')
    return board


# ----------------------------------------------------------------------------- final test (once)
def _block_bootstrap(y, p_by_model: dict, clusters, n_boot=1000, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    uc, inv = np.unique(np.asarray(clusters), return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uc))]
    stats = {k: {'average_precision': [], 'roc_auc': [], 'brier': []} for k in p_by_model}
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uc), len(uc))])
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        for k, p in p_by_model.items():
            pp = np.asarray(p)[idx]
            stats[k]['average_precision'].append(average_precision_score(yy, pp))
            stats[k]['roc_auc'].append(roc_auc_score(yy, pp))
            stats[k]['brier'].append(brier_score_loss(yy, pp))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None  # noqa: E731
    out = {k: {m: ci(v) for m, v in s.items()} for k, s in stats.items()}
    keys = list(p_by_model)
    for a in keys[1:]:
        d = np.array(stats[keys[0]]['average_precision']) - np.array(stats[a]['average_precision'])
        out[f'delta_AP_{keys[0]}_minus_{a}'] = {'ci95': ci(list(d)), 'share_boot_gt_0': float((d > 0).mean())}
    out['_clusters'] = int(len(uc))
    out['_replicates'] = int(n_boot)
    return out


def run_final_test(df: pd.DataFrame, outdir: Path, log=print, allow_rerun: bool = False) -> dict:
    outdir = Path(outdir)
    result_path = outdir / 'test_results.json'
    if result_path.exists() and not allow_rerun:
        raise RuntimeError(f'{result_path} exists: the final test has already been run once. '
                           'Re-running would turn the test set into a development set.')
    bundle = joblib.load(outdir / 'model.joblib')
    frozen = joblib.load(outdir / 'frozen_candidates.joblib')
    reg_models = joblib.load(outdir / 'regression_frozen.joblib') if (outdir / 'regression_frozen.joblib').exists() else {}
    meta = bundle['metadata']
    df = df[df[TARGET].notna()].copy()
    if _hash_frame(df) != meta['dataset_sha256_labelled_rows']:
        raise RuntimeError('Dataset differs from the one used for development; refusing to evaluate.')
    df['split'] = assign_splits(df)
    te, va = df[df.split == 'test'].copy(), df[df.split == 'validation']
    yte = te[TARGET].astype(int).to_numpy()
    feats = bundle['features']

    p_champ = bundle['calibrator'].predict(bundle['model'].predict_proba(te[feats])[:, 1])
    champ = classification_metrics(yte, p_champ, bundle['threshold'])
    champ['reliability'] = reliability_table(yte, p_champ)

    board = []
    raw_preds = {}
    for key, fz in frozen.items():
        model, fs = key.split('|')
        f = FEATURE_SETS.get(fs, WEATHER_FEATURES)
        p = fz['pipeline'].predict_proba(te[f])[:, 1]
        raw_preds[key] = p
        board.append({'model': model, 'feature_set': fs,
                      **{f'test_{k}': v for k, v in classification_metrics(yte, p, fz['threshold_raw']).items()
                         if k != 'confusion_matrix'}})
    board = pd.DataFrame(board).sort_values('test_average_precision', ascending=False)
    board.to_csv(outdir / 'test_leaderboard_frozen_candidates.csv', index=False)

    clim_key = 'climatology_cell_month|n/a (baseline)'
    # calibrated comparators (validation-fit isotonic) for a like-for-like Brier comparison
    yva = va[TARGET].astype(int).to_numpy()
    comps, thresholds = {'champion': p_champ}, {'champion': bundle['threshold']}
    for label, key in [('climatology', clim_key), ('matched_ablation', meta['matched_ablation_partner'])]:
        if key and key in frozen:
            f = FEATURE_SETS.get(key.split('|')[1], WEATHER_FEATURES)
            pv = frozen[key]['pipeline'].predict_proba(va[f])[:, 1]
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(pv, yva)
            comps[label] = iso.predict(raw_preds[key])
            thresholds[label] = best_f1_threshold(yva, iso.predict(pv))   # validation-only threshold
    comp_metrics = {k: classification_metrics(yte, v, thresholds[k]) for k, v in comps.items()}

    weeks = pd.to_datetime(te['target_date']).dt.strftime('%G-W%V').to_numpy()
    boot_time = _block_bootstrap(yte, comps, weeks)
    boot_space = _block_bootstrap(yte, comps, te['cell_id'].to_numpy())

    # sensitivity analyses (no tuning; reporting only)
    sens = {}
    masks = {
        'exclude_cloud_obscured_negatives': ~(te['target_cloud_obscured_proxy'].astype(bool) & (yte == 0)),
        'only_rows_without_issue_day_detection': ~te['detection_on_issue_day'].astype(bool),
        'only_rows_with_issue_day_detection': te['detection_on_issue_day'].astype(bool),
    }
    for name, m in masks.items():
        m = np.asarray(m)
        if m.sum() and len(np.unique(yte[m])) == 2:
            sens[name] = {k: classification_metrics(yte[m], np.asarray(v)[m], bundle['threshold'])
                          for k, v in comps.items()}
    by_year = {}
    for yr, g in te.groupby(pd.to_datetime(te['target_date']).dt.year):
        idx = te.index.get_indexer(g.index)
        by_year[int(yr)] = classification_metrics(yte[idx], p_champ[idx], bundle['threshold'])

    # spatial hold-out: retrain frozen champion configuration without block k
    spatial = {}
    blocks = spatial_block(df)
    cname, cparams = meta['champion']['model'], meta['champion']['params']
    for b in sorted(blocks.unique()):
        trk = df[(df.split == 'train') & (blocks != b)]
        vak = df[(df.split == 'validation') & (blocks != b)]
        tek = df[(df.split == 'test') & (blocks == b)]
        if tek[TARGET].sum() == 0:
            spatial[b] = {'note': 'no positives in held-out block test rows'}
            continue
        pipe = make_pipeline(cname, feats, cparams).fit(trk[feats], trk[TARGET].astype(int))
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(
            pipe.predict_proba(vak[feats])[:, 1], vak[TARGET].astype(int))
        thr = best_f1_threshold(vak[TARGET].astype(int), iso.predict(pipe.predict_proba(vak[feats])[:, 1]))
        ph = iso.predict(pipe.predict_proba(tek[feats])[:, 1])
        clim = Climatology().fit(trk[WEATHER_FEATURES], trk[TARGET].astype(int))
        idx = te.index.get_indexer(tek.index)
        spatial[b] = {'held_out_block': b, 'train_rows': int(len(trk)),
                      'held_out_model': classification_metrics(tek[TARGET].astype(int), ph, thr),
                      'in_domain_champion_same_rows': classification_metrics(yte[idx], p_champ[idx], bundle['threshold']),
                      'climatology_without_block (month-rate fallback)': classification_metrics(
                          tek[TARGET].astype(int), clim.predict_proba(tek[WEATHER_FEATURES])[:, 1], 0.5)}
        log(f"  spatial block {b}: held-out AP={spatial[b]['held_out_model']['average_precision']:.4f} "
            f"in-domain AP={spatial[b]['in_domain_champion_same_rows']['average_precision']:.4f}")

    # predictions + error analysis
    pred = te[['cell_id', 'issue_date', 'target_date', 'latitude', 'longitude', TARGET,
               'target_nextday_count', 'target_cloud_obscured_proxy', 'detection_on_issue_day']].copy()
    pred['probability'] = p_champ
    pred['predicted_label'] = (p_champ >= bundle['threshold']).astype(int)
    pred.to_parquet(outdir / 'test_predictions.parquet', index=False)
    show = ['cell_id', 'target_date', 'probability', 'target_nextday_count', 'target_cloud_obscured_proxy',
            'detection_on_issue_day']
    fps = pred[pred[TARGET] == 0].nlargest(15, 'probability')[show]
    misses = pred[(pred[TARGET] == 1) & (pred['predicted_label'] == 0)].sort_values(
        'target_nextday_count', ascending=False).head(15)[show]
    fn_month = pred[pred[TARGET] == 1].assign(month=pd.to_datetime(pred['target_date']).dt.month) \
        .groupby('month')['predicted_label'].agg(['size', 'mean']).rename(columns={'size': 'positives', 'mean': 'recall'})
    error = {'top_false_positives': fps.to_dict(orient='records'),
             'largest_missed_detections': misses.to_dict(orient='records'),
             'recall_by_target_month': fn_month.reset_index().to_dict(orient='records'),
             'missed_positive_share_with_issue_day_detection': float(
                 pred[(pred[TARGET] == 1) & (pred.predicted_label == 0)]['detection_on_issue_day'].mean())}

    reg = {}
    if reg_models:
        yreg = np.log1p(te['target_nextday_count'].astype(float))
        reg = {n: regression_metrics(yreg, m.predict(te[FEATURES])) for n, m in reg_models.items()}

    results = {'evaluated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
               'champion_key': meta['champion']['key'], 'champion_test': champ,
               'calibrated_comparators_test': comp_metrics,
               'bootstrap_ci95_week_clusters': boot_time, 'bootstrap_ci95_cell_clusters': boot_space,
               'test_by_target_year': by_year, 'sensitivity': sens, 'spatial_holdout': spatial,
               'error_analysis': error, 'regression_log1p_count_test': reg,
               'note': 'Test split evaluated once after freezing; see frozen_champion.json for the timestamp.'}
    result_path.write_text(json.dumps(results, indent=2, default=str) + '\n')
    meta['test_evaluated'] = True
    meta['test_average_precision'] = champ['average_precision']
    meta['test_prevalence'] = champ['prevalence']
    bundle['metadata'] = meta
    joblib.dump(bundle, outdir / 'model.joblib')
    return results
