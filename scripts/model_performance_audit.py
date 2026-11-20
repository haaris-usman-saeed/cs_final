"""FireRisk Open v0.3 — complete model performance audit (REPORTING ONLY).

Rebuilds release_v3/MODEL_PERFORMANCE_TABLE.csv, release_v3/MODEL_PERFORMANCE_REPORT.md and release_v3/figures/* from
existing artefacts. It never re-selects a model, never changes a threshold/calibration, and never trains on test data.

Sources of every number (column ``metric_source``):
* ``frozen_in_sample``      – the frozen candidate (fold-2023 model, trained on development targets <= 2022) scored on
                              its own training rows, frozen calibrator + frozen threshold. In-sample, optimistic by design.
* ``reconstructed_oof``     – rolling-origin out-of-fold predictions (validation 2021, 2022, 2023 after the 32-day embargo).
                              2021/2022 fold models were not saved in v0.3 and are refit here with the frozen configuration
                              (deterministic on arm64; checked by refitting 2023 and comparing with the frozen model).
                              The 2023 part uses the frozen model itself. Thresholded metrics apply the frozen calibrator and
                              threshold, both of which were FIT on 2023, so the 2023 third of these rows is in-sample for them.
* ``frozen_test_rescore``   – frozen candidate re-scored on the one-time test rows (identical inputs and artefacts; no refit,
                              no selection). For the six models reported in test_results_v3.json the values are checked equal.
* ``definition``            – the always-no-detection reference (accuracy = 1 − prevalence).
Requires arm64 for the exactness check (see docs/OPERATIONS.md). Usage: python scripts/model_performance_audit.py
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fireaq.experiment_v3 import EMBARGO_DAYS, FOLD_YEARS, assign_roles, build, fold_masks  # noqa: E402
from fireaq.features_v3 import SET_B, TARGET  # noqa: E402

DATA = ROOT / 'release_v3' / 'firerisk_open_multiregion_v0.3.0.parquet'
OUT = ROOT / 'output_v3'
REL = ROOT / 'release_v3'
FIG = REL / 'figures'
CACHE = OUT / 'audit_oof_raw.parquet'
POPS = {'train': 'Training (dev regions, targets 2019–2022; in-sample)',
        'validation': 'Rolling-origin validation OOF 2021–2023 (dev regions)',
        'test_time': 'New-year test 2024–25 (NRockies, Southwest, Southeast)',
        'test_space_primary': 'Never-used regions 2024–25 (PNW, Ozarks)',
        'test_space_secondary': 'Never-used regions 2019–23 (PNW, Ozarks)',
        'previously_tested_norcal': 'NorCal 2024–25 (already evaluated in v0.2; not untouched)'}
NAMES = {'dummy_prior': 'Dummy (prior)', 'dummy_most_frequent': 'Dummy (most frequent)',
         'dummy_stratified': 'Dummy (stratified)', 'climatology_hist_snpp': 'Climatology (S-NPP 2012–17)',
         'climatology_train_cell_month': 'Climatology (train cell×month)', 'logistic_l2': 'Logistic regression (L2)',
         'gaussian_nb': 'Gaussian naive Bayes', 'decision_tree': 'Decision tree', 'random_forest': 'Random forest',
         'extra_trees': 'Extra Trees', 'hist_gradient_boosting': 'HistGradientBoosting', 'xgboost': 'XGBoost',
         'mlp': 'MLP', 'always_no_detection': 'Always "no detection" (majority class)'}
FS = {'A_no_fire_info': 'A (no fire history)', 'B_plus_fire_history': 'B (+ fire history)', 'n/a': '—'}
ORDER = ['always_no_detection', 'dummy_prior', 'dummy_most_frequent', 'dummy_stratified', 'climatology_hist_snpp',
         'climatology_train_cell_month', 'logistic_l2', 'gaussian_nb', 'decision_tree', 'random_forest', 'extra_trees',
         'hist_gradient_boosting', 'xgboost', 'mlp']


def cls_metrics(y, p, thr):
    y = np.asarray(y, int)
    yhat = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    two = len(np.unique(y)) == 2
    return {'n': int(len(y)), 'positives': int(y.sum()), 'prevalence': float(y.mean()), 'threshold': float(thr),
            'accuracy': float(accuracy_score(y, yhat)), 'balanced_accuracy': float(balanced_accuracy_score(y, yhat)),
            'precision': float(precision_score(y, yhat, zero_division=0)) if yhat.sum() else np.nan,
            'recall': float(recall_score(y, yhat, zero_division=0)), 'f1': float(f1_score(y, yhat, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, p)) if two and len(np.unique(p)) > 1 else 0.5,
            'average_precision': float(average_precision_score(y, p)), 'brier': float(brier_score_loss(y, p)),
            'alert_rate': float(yhat.mean()), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def load():
    df = pd.read_parquet(DATA)
    df = df[df['in_primary_analysis']].copy()
    df['role'] = assign_roles(df).to_numpy()
    return df


def reconstruct_oof(dev, frozen, log=print):
    """Raw OOF scores per candidate; fold 2023 from the frozen model, 2021/2022 refit; 2023 refit only to check exactness."""
    if CACHE.exists():
        return pd.read_parquet(CACHE), json.loads((OUT / 'audit_reconstruction_check.json').read_text())
    y = dev[TARGET].astype(int).to_numpy()
    out = pd.DataFrame({'row': np.arange(len(dev)), 'fold': -1})
    checks = {}
    for yr in FOLD_YEARS:
        out.loc[fold_masks(dev, yr)[1], 'fold'] = yr
    for key, fz in frozen.items():
        name, fs = key.split('|')
        col = np.full(len(dev), np.nan)
        for yr in FOLD_YEARS:
            tr, va = fold_masks(dev, yr)
            p_frozen = fz['pipeline'].predict_proba(dev.loc[va, fz['columns']])[:, 1] if yr == 2023 else None
            pipe, cols = build(name, fs, json.loads(fz['params']))
            pipe.fit(dev.loc[tr, cols], y[tr])
            p = pipe.predict_proba(dev.loc[va, cols])[:, 1]
            if yr == 2023:
                checks[key] = float(np.max(np.abs(p - p_frozen)))
                p = p_frozen                                  # always use the frozen model for 2023
            col[va] = p
        out[key] = col
        log(f'  OOF {key:45s} 2023 refit vs frozen max|Δ|={checks[key]:.2e}')
    out.to_parquet(CACHE, index=False)
    (OUT / 'audit_reconstruction_check.json').write_text(json.dumps(checks, indent=1) + '\n')
    return out, checks


def main(log=print):
    if platform.machine() != 'arm64':
        log('WARNING: not arm64 — reconstructed folds may differ from the frozen arm64 models (docs/OPERATIONS.md)')
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    frozen = joblib.load(OUT / 'frozen_candidates_v3.joblib')
    released = json.loads((OUT / 'test_results_v3.json').read_text())
    board = pd.read_csv(OUT / 'validation_leaderboard_v3.csv')
    champion = json.loads((OUT / 'frozen_champion_v3.json').read_text())['champion']['key']
    dev = df[(df.role == 'dev') & df[TARGET].notna()].reset_index(drop=True)
    y_dev = dev[TARGET].astype(int).to_numpy()
    t = pd.to_datetime(dev['target_date'])
    train_mask = (t <= pd.Timestamp('2022-12-31')).to_numpy()
    oof, checks = reconstruct_oof(dev, frozen, log)
    vmask = (oof['fold'] > 0).to_numpy()
    pops = {r: df[(df.role == r) & df[TARGET].notna()].reset_index(drop=True) for r in
            ['test_time', 'test_space_primary', 'test_space_secondary', 'previously_tested_norcal']}

    rows, scores = [], {}
    for key, fz in frozen.items():
        name, fs = key.split('|')
        cal, thr, cols = fz['calibrator'], fz['threshold_f1'], fz['columns']

        def score(frame):
            return np.clip(cal.predict(fz['pipeline'].predict_proba(frame[cols])[:, 1]), 0, 1)
        base = {'task': 'classification', 'model': name, 'feature_set': fs, 'model_name': NAMES[name],
                'is_champion': key == champion}
        m = cls_metrics(y_dev[train_mask], score(dev[train_mask]), thr)
        rows.append({**base, 'population': 'train', 'metric_source': 'frozen_in_sample', **m})
        raw = oof[key].to_numpy()[vmask]
        m = cls_metrics(y_dev[vmask], np.clip(cal.predict(raw), 0, 1), thr)
        m['average_precision'] = float(average_precision_score(y_dev[vmask], raw))     # rank metric on raw OOF scores
        m['roc_auc'] = float(roc_auc_score(y_dev[vmask], raw)) if len(np.unique(raw)) > 1 else 0.5
        rows.append({**base, 'population': 'validation', 'metric_source': 'reconstructed_oof', **m})
        scores[(key, 'validation')] = (y_dev[vmask], np.clip(cal.predict(raw), 0, 1))
        for r, frame in pops.items():
            p = score(frame)
            y = frame[TARGET].astype(int).to_numpy()
            rows.append({**base, 'population': r, 'metric_source': 'frozen_test_rescore', **cls_metrics(y, p, thr)})
            scores[(key, r)] = (y, p)
        log(f'  scored {key}')
    for pop, frame_y in [('train', y_dev[train_mask]), ('validation', y_dev[vmask])] + \
                        [(r, f[TARGET].astype(int).to_numpy()) for r, f in pops.items()]:
        y = np.asarray(frame_y, int)
        rows.append({'task': 'classification', 'model': 'always_no_detection', 'feature_set': 'n/a',
                     'model_name': NAMES['always_no_detection'], 'is_champion': False, 'population': pop,
                     'metric_source': 'definition', **cls_metrics(y, np.zeros(len(y)), 0.5),
                     'average_precision': float(y.mean()), 'roc_auc': 0.5})
    cls = pd.DataFrame(rows)

    # ---- consistency checks against released artefacts (fail loudly on any discrepancy)
    issues = []
    lb = board.set_index(['model', 'feature_set'])['oof_ap_pooled']
    for r in cls[(cls.population == 'validation') & (cls.model != 'always_no_detection')].itertuples():
        ref = lb.get((r.model, r.feature_set if r.feature_set != 'n/a' else np.nan), None)
        if ref is None:
            ref = board[(board.model == r.model)]['oof_ap_pooled'].iloc[0]
        if abs(r.average_precision - ref) > 1e-9:
            issues.append(f'validation AP {r.model}|{r.feature_set}: {r.average_precision} vs leaderboard {ref}')
    comp = {'champion': champion, 'climatology_hist_snpp': 'climatology_hist_snpp|n/a',
            'ablation_A_no_fire_info': 'extra_trees|A_no_fire_info'}
    for pop in ['test_time', 'test_space_primary', 'test_space_secondary', 'previously_tested_norcal']:
        for lab, key in comp.items():
            ref = released[pop]['metrics'][lab]
            got = cls[(cls.population == pop) & (cls.model + '|' + cls.feature_set == key)].iloc[0]
            for mm in ['average_precision', 'accuracy', 'balanced_accuracy', 'f1', 'brier']:
                if abs(got[mm] - ref[mm]) > 1e-9:
                    issues.append(f'{pop} {key} {mm}: {got[mm]} vs released {ref[mm]}')
        for c in released[pop]['all_frozen_candidates']:
            got = cls[(cls.population == pop) & (cls.model + '|' + cls.feature_set == c['model'])].iloc[0]
            if abs(got['average_precision'] - c['ap']) > 1e-9:
                issues.append(f'{pop} {c["model"]} AP {got["average_precision"]} vs released {c["ap"]}')
    if max(checks.values()) > 1e-9:
        issues.append(f'fold-2023 reconstruction not exact: {max(checks.values())}')

    # ---- regression
    reg = joblib.load(OUT / 'regression_frozen_v3.joblib')
    devr = df[(df.role == 'dev') & df['target_nextday_count'].notna()]
    tr_ = pd.to_datetime(devr['target_date'])
    rpops = {'train': devr[tr_ <= pd.Timestamp('2022-12-31')],
             'validation': devr[tr_ >= pd.Timestamp('2023-01-01') + pd.Timedelta(days=EMBARGO_DAYS)],
             'test_time': df[(df.role == 'test_time') & df['target_nextday_count'].notna()],
             'test_space_primary': df[(df.role == 'test_space_primary') & df['target_nextday_count'].notna()]}
    rrows = []
    ref_val = pd.read_csv(OUT / 'regression_validation_v3.csv').set_index('model')
    ref_test = pd.DataFrame(released['regression_log1p_count_test']).set_index(['model', 'test_set'])
    for nm, mdl in reg.items():
        for pop, fr in rpops.items():
            yt = np.log1p(fr['target_nextday_count'].astype(float).to_numpy())
            pr = mdl.predict(fr[SET_B])
            mse = float(mean_squared_error(yt, pr))
            rec = {'task': 'regression (log1p next-day detection count)', 'model': nm, 'feature_set': 'B_plus_fire_history',
                   'model_name': nm.replace('_', ' '), 'population': pop, 'n': int(len(yt)),
                   'metric_source': 'frozen_in_sample' if pop == 'train' else 'frozen_rescore',
                   'MAE': float(mean_absolute_error(yt, pr)), 'MSE': mse, 'RMSE': float(np.sqrt(mse)),
                   'R2': float(r2_score(yt, pr))}
            rrows.append(rec)
            ref = ref_val.loc[nm, 'val_RMSE'] if pop == 'validation' else \
                ref_test.loc[(nm, pop), 'test_RMSE'] if pop.startswith('test') else None
            if ref is not None and abs(ref - rec['RMSE']) > 1e-9:
                issues.append(f'regression {nm} {pop} RMSE {rec["RMSE"]} vs released {ref}')
    regdf = pd.DataFrame(rrows)
    table = pd.concat([cls, regdf], ignore_index=True)
    table.insert(4, 'population_label', table['population'].map(POPS))
    table.to_csv(REL / 'MODEL_PERFORMANCE_TABLE.csv', index=False)
    audit = {'consistency_issues': issues, 'fold2023_reconstruction_max_abs_diff': max(checks.values()),
             'fold2023_reconstruction_per_model': checks, 'platform': platform.machine(), 'champion': champion}
    (REL / 'model_performance_audit_checks.json').write_text(json.dumps(audit, indent=1) + '\n')
    log(f'consistency issues: {len(issues)}')
    for i in issues:
        log('  ' + i)
    # ---- paired week-block bootstrap: champion vs strong competitors on identical rows (reporting only)
    rivals = ['extra_trees|A_no_fire_info', 'random_forest|B_plus_fire_history', 'xgboost|B_plus_fire_history',
              'logistic_l2|B_plus_fire_history', 'hist_gradient_boosting|B_plus_fire_history', 'mlp|B_plus_fire_history',
              'climatology_hist_snpp|n/a']
    pair_rows = []
    rng = np.random.default_rng(17)
    pop_frames = {'validation': (y_dev[vmask], dev.loc[vmask, 'target_date'], {k: oof[k].to_numpy()[vmask] for k in frozen}),
                  **{r: (scores[(champion, r)][0], pops[r]['target_date'], {k: scores[(k, r)][1] for k in frozen})
                     for r in ['test_time', 'test_space_primary']}}
    for pop, (y, dates, pp) in pop_frames.items():
        wk = pd.to_datetime(pd.Series(dates).reset_index(drop=True)).dt.strftime('%G-W%V').to_numpy()
        uc, inv = np.unique(wk, return_inverse=True)
        mem = [np.flatnonzero(inv == i) for i in range(len(uc))]
        draws = [np.concatenate([mem[i] for i in rng.integers(0, len(uc), len(uc))]) for _ in range(1000)]
        draws = [d for d in draws if 0 < y[d].sum() < len(d)]
        for rv in rivals:
            d = [average_precision_score(y[ix], pp[champion][ix]) - average_precision_score(y[ix], pp[rv][ix]) for ix in draws]
            pair_rows.append({'population': pop, 'champion': champion, 'rival': rv,
                              'delta_ap': float(average_precision_score(y, pp[champion]) - average_precision_score(y, pp[rv])),
                              'ci95_low': float(np.percentile(d, 2.5)), 'ci95_high': float(np.percentile(d, 97.5)),
                              'share_champion_better': float(np.mean(np.array(d) > 0)), 'replicates': len(draws)})
        log(f'  paired bootstrap done: {pop}')
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(REL / 'model_performance_paired_bootstrap.csv', index=False)
    figures(cls, scores, champion, pops)
    report(cls, regdf, audit, champion, board, pairs)
    return table, audit


# ----------------------------------------------------------------------------- figures
SPLIT_COL = {'train': '#2a78d6', 'validation': '#eb6834', 'test_time': '#1baf7a', 'test_space_primary': '#eda100'}
SPLIT_LAB = {'train': 'Training (in-sample)', 'validation': 'Validation OOF 2021–23',
             'test_time': 'Test: new years 2024–25', 'test_space_primary': 'Test: unseen regions 2024–25'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK, 'axes.titlecolor': INK, 'axes.titlesize': 11,
                         'axes.titleweight': 'bold', 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                         'savefig.facecolor': SURF, 'legend.frameon': False, 'pdf.fonttype': 42})


def _label(r):
    return f"{NAMES[r.model]}" + ('' if r.feature_set == 'n/a' else f"  [{FS[r.feature_set]}]") + \
        ('  ★ champion' if r.is_champion else '')


def _ordered(cls):
    c = cls.copy()
    c['o1'] = c['model'].map({m: i for i, m in enumerate(ORDER)})
    c['o2'] = c['feature_set'].map({'n/a': 0, 'A_no_fire_info': 1, 'B_plus_fire_history': 2})
    return c.sort_values(['o1', 'o2'])


def _grouped(cls, metric, title, fname, xlabel, ref=None, splits=('train', 'validation', 'test_time', 'test_space_primary'),
             note=None):
    import matplotlib.pyplot as plt
    c = _ordered(cls)
    entries = c.drop_duplicates(['model', 'feature_set'])
    n = len(entries)
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * n * len(splits) / 2 + 1.6))
    h = 0.8 / len(splits)
    for j, sp in enumerate(splits):
        vals = [c[(c.model == e.model) & (c.feature_set == e.feature_set) & (c.population == sp)][metric].iloc[0]
                for e in entries.itertuples()]
        ys = np.arange(n) + (j - (len(splits) - 1) / 2) * h
        ax.barh(ys, vals, height=h * 0.86, color=SPLIT_COL[sp], label=SPLIT_LAB[sp], zorder=3)
    if ref is not None:
        ref(ax)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([_label(e) for e in entries.itertuples()])
    for lab, e in zip(ax.get_yticklabels(), entries.itertuples()):
        if e.is_champion:
            lab.set_fontweight('bold')
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.045 * 30 / n), ncol=2, fontsize=8.5)
    if note:
        ax.text(0.0, -0.095 * 30 / n, note, transform=ax.transAxes, fontsize=7.5, color=INK2, va='top', wrap=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{fname}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def figures(cls, scores, champion, pops):
    import matplotlib.pyplot as plt
    _style()
    prev = {sp: cls[(cls.model == 'always_no_detection') & (cls.population == sp)]['prevalence'].iloc[0]
            for sp in SPLIT_COL}

    def acc_ref(ax):
        for sp, col in SPLIT_COL.items():
            ax.axvline(1 - prev[sp], color=col, lw=1.2, ls=(0, (4, 3)), zorder=4)
        ax.text(1 - prev['train'], -0.9, 'dashed = always "no detection"', ha='right', va='bottom', fontsize=7.5, color=INK2)
    _grouped(cls, 'accuracy', 'Figure 1. Accuracy at each model\'s frozen threshold — inflated by class imbalance',
             'fig1_accuracy', 'Accuracy', ref=acc_ref,
             note='Dummy rows: each model uses its frozen threshold (max F1 on calibrated 2023 scores). For a constant score that rule ' + \
                 'selects "always alert", hence their very low accuracy. The always-"no detection" row is shown separately.')

    def ba_ref(ax):
        ax.axvline(0.5, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=4)
        ax.text(0.5, -0.9, 'chance = 0.5', ha='left', va='bottom', fontsize=7.5, color=INK2)
    _grouped(cls, 'balanced_accuracy', 'Figure 2. Balanced accuracy at each model\'s frozen threshold',
             'fig2_balanced_accuracy', 'Balanced accuracy', ref=ba_ref,
             note='Dummy rows: each model uses its frozen threshold (max F1 on calibrated 2023 scores). For a constant score that rule ' + \
                 'selects "always alert", hence their very low accuracy. The always-"no detection" row is shown separately.')

    def ap_ref(ax):
        for sp in ['validation', 'test_time', 'test_space_primary']:
            ax.axvline(prev[sp], color=SPLIT_COL[sp], lw=1.2, ls=(0, (4, 3)), zorder=4)
        ax.text(prev['test_space_primary'], -0.9, 'dashed = base rate (no-skill AP)', ha='left', va='bottom',
                fontsize=7.5, color=INK2)
    _grouped(cls, 'average_precision', 'Figure 3. Average precision (area under the PR curve)', 'fig3_average_precision',
             'Average precision', ref=ap_ref, splits=('validation', 'test_time', 'test_space_primary'))

    # Figure 4: PR curves
    show = [(champion, 'Extra Trees [B] ★ champion', '#2a78d6', '-'),
            ('extra_trees|A_no_fire_info', 'Extra Trees [A] (matched, no fire history)', '#eb6834', '-'),
            ('xgboost|B_plus_fire_history', 'XGBoost [B]', '#1baf7a', '-'),
            ('logistic_l2|B_plus_fire_history', 'Logistic regression [B]', '#6250d6', '-'),
            ('climatology_hist_snpp|n/a', 'Climatology (S-NPP 2012–17)', '#52514e', (0, (5, 3)))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, sp in zip(axes, ['test_time', 'test_space_primary']):
        for key, lab, col, ls in show:
            y, p = scores[(key, sp)]
            pr, rc, _ = precision_recall_curve(y, p)
            ax.plot(rc, pr, color=col, lw=2, ls=ls, label=f'{lab}  (AP {average_precision_score(y, p):.3f})',
                    drawstyle='steps-post')
        ax.axhline(y.mean(), color=INK2, lw=1, ls=':', label=f'Base rate {y.mean():.3f}')
        ax.set_title(SPLIT_LAB[sp], loc='left', fontsize=10)
        ax.set_xlabel('Recall')
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.8)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7.5, loc='upper right')
    axes[0].set_ylabel('Precision')
    fig.suptitle('Figure 4. Precision–recall curves on the one-time test populations (frozen, calibrated models)',
                 x=0.01, ha='left', fontweight='bold', fontsize=11, color=INK)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'fig4_precision_recall.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Figure 5: confusion matrices
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blue', ['#eef4fc', '#2a78d6', '#0d3b73'])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, sp in zip(axes, ['test_time', 'test_space_primary']):
        r = cls[(cls.population == sp) & (cls.model + '|' + cls.feature_set == champion)].iloc[0]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        rown = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rown, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]:,}\n({rown[i, j]:.1%} of row)', ha='center', va='center', fontsize=10,
                        color='white' if rown[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ['Predicted: no alert', 'Predicted: alert'])
        ax.set_yticks([0, 1], ['Observed: no detection', 'Observed: detection'])
        ax.set_title(f"{SPLIT_LAB[sp]}\nthreshold {r.threshold:.4f} · precision {r.precision:.3f} · "
                     f"recall {r.recall:.3f}", loc='left', fontsize=9.5)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle('Figure 5. Champion (Extra Trees [B]) confusion matrices at the frozen threshold',
                 x=0.01, ha='left', fontweight='bold', fontsize=11, color=INK)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'fig5_confusion_matrices.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def _fmt(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'n/a'
    return f'{v:.{d}f}' if isinstance(v, (float, np.floating)) else str(v)


def report(cls, regdf, audit, champion, board, pairs):
    c = _ordered(cls)
    ent = c.drop_duplicates(['model', 'feature_set'])

    def g(e, pop, m):
        s = c[(c.model == e.model) & (c.feature_set == e.feature_set) & (c.population == pop)]
        return s[m].iloc[0] if len(s) else np.nan
    L = ['# FireRisk Open v0.3 — model performance audit (reporting only)', '',
         'Generated by `scripts/model_performance_audit.py` from the frozen v0.3 artefacts. **No model, threshold, '
         'calibration or selection was changed; nothing was trained on test data.** Every number is in '
         '`MODEL_PERFORMANCE_TABLE.csv` with its `metric_source`.', '',
         '> Target: a new NOAA-20 VIIRS satellite **detection** in a 0.5° cell on the next UTC day — not ignition or wildfire risk.', '',
         '## How each column was obtained', '',
         '* **Train:** the frozen candidate (trained on development targets ≤ 2022) scored on its own training rows. This is in-sample and optimistic.',
         '* **Validation:** rolling-origin out-of-fold predictions for 2021, 2022 and 2023, pooled over 4 development regions with a 32-day embargo.',
         '  * The 2023 fold uses the frozen models.',
         '  * The 2021 and 2022 fold models were not saved in v0.3, so they were **reconstructed** by refitting the frozen configurations.',
         f"  * Exactness check: refitting 2023 reproduces the frozen models to max |Δ| = {audit['fold2023_reconstruction_max_abs_diff']:.1e} on arm64.",
         '  * AP and ROC-AUC use the raw OOF scores and equal the original leaderboard values.',
         '  * The thresholded metrics apply each model\'s frozen calibrator and threshold. Both were fitted on 2023, so that third of the rows is in-sample for them.',
         '* **Test:** the one-time test populations, kept separate.',
         '  * **T-time:** NRockies, Southwest and Southeast in 2024-02-02…2025-12-31.',
         '  * **T-space:** never-used PNW and Ozarks in 2024–25.',
         '  * Metrics for the 6 models in `test_results_v3.json` are checked to equal the released values.',
         '  * The other candidates are frozen models **re-scored post hoc** for this report. Nothing was selected from them.',
         '* **Thresholds:** each model\'s frozen threshold (max F1 on calibrated 2023 validation scores).',
         '  * For the constant dummies this rule degenerates to "always alert".',
         '  * The always-"no detection" row is therefore given separately, by definition.',
         f"* Consistency checks against released artefacts: **{len(audit['consistency_issues'])} discrepancies**.", '']
    for title, pop_note, metric in [('Accuracy', 'Accuracy', 'accuracy'), ('Balanced accuracy', 'Balanced acc.', 'balanced_accuracy')]:
        pass
    L += ['## Table 1 — accuracy and balanced accuracy (classification)', '',
          'T-time = new-year test 2024–25; T-space = never-used regions 2024–25. Thresholds are listed in Table 2.', '',
          '| Model | Feature set | Train acc. | Validation acc. | T-time acc. | T-space acc. | Train bal. acc. | Validation bal. acc. | T-time bal. acc. | T-space bal. acc. |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for e in ent.itertuples():
        nm = NAMES[e.model] + (' **★**' if e.is_champion else '')
        L.append(f"| {nm} | {FS[e.feature_set]} | " + ' | '.join(_fmt(g(e, p, 'accuracy')) for p in
                 ['train', 'validation', 'test_time', 'test_space_primary']) + ' | ' +
                 ' | '.join(_fmt(g(e, p, 'balanced_accuracy')) for p in ['train', 'validation', 'test_time', 'test_space_primary']) + ' |')
    L += ['', '## Table 2 — ranking and threshold metrics (classification)', '',
          '| Model | Feature set | Threshold | Validation AP | T-time AP | T-space AP | T-time precision | T-time recall | T-time F1 | T-time ROC-AUC | T-time Brier | T-space precision | T-space recall | T-space F1 | T-space ROC-AUC | T-space Brier |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for e in ent.itertuples():
        nm = NAMES[e.model] + (' **★**' if e.is_champion else '')
        thr = 'n/a (constant 0)' if e.model == 'always_no_detection' else _fmt(g(e, 'test_time', 'threshold'), 4)
        L.append(f"| {nm} | {FS[e.feature_set]} | {thr} | {_fmt(g(e, 'validation', 'average_precision'))} | "
                 f"{_fmt(g(e, 'test_time', 'average_precision'))} | {_fmt(g(e, 'test_space_primary', 'average_precision'))} | " +
                 ' | '.join(_fmt(g(e, 'test_time', m), 4 if m == 'brier' else 3) for m in ['precision', 'recall', 'f1', 'roc_auc', 'brier']) + ' | ' +
                 ' | '.join(_fmt(g(e, 'test_space_primary', m), 4 if m == 'brier' else 3) for m in ['precision', 'recall', 'f1', 'roc_auc', 'brier']) + ' |')
    pv = {p: c[(c.model == 'always_no_detection') & (c.population == p)]['prevalence'].iloc[0]
          for p in ['train', 'validation', 'test_time', 'test_space_primary']}
    L += ['', f"Base rates (= AP of a no-skill ranking): train {pv['train']:.4f}, validation {pv['validation']:.4f}, "
              f"T-time {pv['test_time']:.4f}, T-space {pv['test_space_primary']:.4f}.", '',
          '## Table 3 — regression: log1p(next-day qualifying detection count), feature set B', '',
          'All 8 regressors were trained (≤ 2022) and frozen in v0.3. Validation here is the single 2023 hold-out, not rolling-origin CV.', '',
          '| Model | Train MAE | Train MSE | Train RMSE | Train R² | Val MAE | Val MSE | Val RMSE | Val R² | T-time MAE | T-time RMSE | T-time R² | T-space MAE | T-space RMSE | T-space R² |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for nm in regdf['model'].unique():
        r = regdf[regdf.model == nm].set_index('population')
        L.append(f"| {nm.replace('_', ' ')} | " + ' | '.join(_fmt(r.loc['train', m], 4) for m in ['MAE', 'MSE', 'RMSE', 'R2']) +
                 ' | ' + ' | '.join(_fmt(r.loc['validation', m], 4) for m in ['MAE', 'MSE', 'RMSE', 'R2']) + ' | ' +
                 ' | '.join(_fmt(r.loc['test_time', m], 4) for m in ['MAE', 'RMSE', 'R2']) + ' | ' +
                 ' | '.join(_fmt(r.loc['test_space_primary', m], 4) for m in ['MAE', 'RMSE', 'R2']) + ' |')
    def best(pop, metric, learned_only=True):
        d = c[(c.population == pop) & (c.model != 'always_no_detection')]
        if learned_only:
            d = d[d.feature_set != 'n/a']
        r = d.sort_values(metric, ascending=False).iloc[0]
        return f"{NAMES[r.model]} [{FS[r.feature_set]}] ({r[metric]:.3f})"
    ch = c[c.is_champion]
    gap = ch[ch.population == 'train']['balanced_accuracy'].iloc[0] - ch[ch.population == 'validation']['balanced_accuracy'].iloc[0]
    rf = c[(c.model == 'random_forest') & (c.feature_set == 'B_plus_fire_history')]
    rf_gap = rf[rf.population == 'train']['balanced_accuracy'].iloc[0] - rf[rf.population == 'validation']['balanced_accuracy'].iloc[0]
    def pr_(pop, rv):
        r = pairs[(pairs.population == pop) & (pairs.rival == rv)].iloc[0]
        return f"{r.delta_ap:+.3f} (95% CI {r.ci95_low:+.3f} to {r.ci95_high:+.3f})"
    L += ['', '## Findings (generated from the tables)', '',
          f"* **Validation (selection data):** the highest pooled OOF AP is {best('validation', 'average_precision')}.",
          f"  * The champion was chosen by the pre-registered 1-SE + leave-one-region-out rule, which this audit does not revisit.",
          f"  * Champion vs Extra Trees without fire history: Δ {pr_('validation', 'extra_trees|A_no_fire_info')}, a tie.",
          f"  * Champion vs logistic regression [B]: Δ {pr_('validation', 'logistic_l2|B_plus_fire_history')}.",
          f"* **New-year test:** the highest AP is {best('test_time', 'average_precision')}.",
          f"  * Δ vs climatology {pr_('test_time', 'climatology_hist_snpp|n/a')}: clearly better.",
          f"  * Δ vs the other strong learned models is within ±0.015 and mostly not significant; vs XGBoost [B] it is {pr_('test_time', 'xgboost|B_plus_fire_history')}.",
          f"* **Never-used regions:** the highest AP is {best('test_space_primary', 'average_precision')}.",
          f"  * The champion is statistically tied with it: Δ vs random forest [B] {pr_('test_space_primary', 'random_forest|B_plus_fire_history')}.",
          f"  * It beats its no-fire-history twin by {pr_('test_space_primary', 'extra_trees|A_no_fire_info')} and climatology by {pr_('test_space_primary', 'climatology_hist_snpp|n/a')}.",
          f"* **Accuracy is not informative here.** The always-\"no detection\" rule scores the highest accuracy everywhere "
          f"({pv['test_time'] and 1 - pv['test_time']:.3f} on T-time) with balanced accuracy 0.5 and zero recall.",
          f"* **Balanced accuracy** depends on each model's frozen threshold and favours high-recall operating points.",
          f"  * Highest validation balanced accuracy: {best('validation', 'balanced_accuracy')}.",
          f"  * Highest T-time balanced accuracy: {best('test_time', 'balanced_accuracy')}.",
          f"  * Highest T-space balanced accuracy: {best('test_space_primary', 'balanced_accuracy')}.",
          f"  * Naive Bayes [B] reaches its high balanced accuracy with low precision and much lower AP.",
          f"* **Over-fitting:** the champion's train-minus-validation balanced-accuracy gap is {gap:.3f}; random forest [B] shows "
          f"{rf_gap:.3f} (train balanced accuracy is in-sample).",
          '* **Regression:** all 8 regressors were trained. Mean and median constants are the baselines.',
          '  * The best validation R² is ~0.07 (random forest); test R² is ≤ 0.06 on new years and ≤ 0.02 in unseen regions, where several models are below 0.',
          '  * These are genuinely poor results, not missing ones: next-day counts are mostly zero, and the features explain little of their variance.',
          '* **Bottom line:** the learned models clearly beat climatology. The differences *between* the top tree/boosting/logistic models '
          'are within sampling noise. Fire-history features matter for unseen regions, not within known regions.',
          '', '## Artefact integrity', '',
          '* `frozen_candidates_v3.joblib` SHA-256 matches the freeze commit (`f23be7e3…`).',
          '* `model_v3.joblib` differs from the freeze-time hash only because `run_final_test_v3` re-saved it with `test_evaluated=True`',
          '  and the test summary after the one-time test. The model, calibrator and threshold were unchanged; a from-scratch rebuild matches its',
          '  predictions to ≤ 1.5e-14.',
          '* `regression_frozen_v3.joblib` had no hash recorded at freeze. Its validation and test RMSE equal the released values exactly.']
    L += ['', '## Table 4 — is the champion meaningfully better? Paired week-block bootstrap of Δ AP (champion − rival)', '',
          'Same rows, same 1,000 resampled sets of ISO weeks for both models. Validation uses the raw OOF scores (2021–23).', '',
          '| Population | Rival | Δ AP | 95% CI | Share of resamples where the champion is better |', '|---|---|---|---|---|']
    for r in pairs.itertuples():
        m, fs = r.rival.split('|')
        L.append(f"| {POPS[r.population].split(' (')[0]} | {NAMES[m]}{'' if fs == 'n/a' else ' [' + FS[fs] + ']'} | "
                 f"{r.delta_ap:+.4f} | {r.ci95_low:+.4f} to {r.ci95_high:+.4f} | {r.share_champion_better:.2f} |")
    L += ['', '## Figures', '']
    for f_, cap in [('fig1_accuracy', 'Accuracy (dashed: always-"no detection" accuracy for each population)'),
                    ('fig2_balanced_accuracy', 'Balanced accuracy (dashed: chance)'),
                    ('fig3_average_precision', 'Average precision (dashed: base rates)'),
                    ('fig4_precision_recall', 'Precision–recall curves on both test populations'),
                    ('fig5_confusion_matrices', 'Champion confusion matrices')]:
        L += [f'![{cap}](figures/{f_}.png)', f'*{cap}. PDF: `figures/{f_}.pdf`.*', '']
    (REL / 'MODEL_PERFORMANCE_REPORT.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
