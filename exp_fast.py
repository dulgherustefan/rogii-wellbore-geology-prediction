"""Fast model experiments on the dumped pipeline-B feature table.

Feature building is ~4 of every 5-hour retrain, so once train_df_B.pkl exists the
model side can be explored in minutes. Reports the same pooled well-level CV the
kernel prints, which is the number that maps to the leaderboard (~1:1 for
non-spatial changes -- see spatialcv.py for why spatial ones do not).

Usage:  python3 exp_fast.py [well_fraction] [config ...]

train_df_B.pkl is written by our own training run, so unpickling it is safe.
"""
import numpy as np, pandas as pd, json, sys, time, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.signal import savgol_filter

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
WANT = set(sys.argv[2:]) or None

t0 = time.time()
df = pd.read_pickle('train_df_B.pkl')
feats = json.load(open('feats_B.json'))
if FRAC < 1.0:
    wells = df['well'].unique()
    keep = set(pd.Series(wells).sample(frac=FRAC, random_state=0))
    df = df[df['well'].isin(keep)].reset_index(drop=True)
print(f'loaded {df.shape} feats={len(feats)} wells={df.well.nunique()} ({time.time()-t0:.0f}s)')

# Pull out every array we need, then drop the frame: the table is ~3.3 GB and
# holding it alongside the training copies is what OOMs this machine.
import gc
y = df['target'].values.astype(np.float32)
last = df['last_known_tvt'].values.astype(float)
y_tvt = last + y
mds = df['md_since'].values.astype(float)
pos = [np.asarray(ix) for _, ix in df.groupby('well', sort=False).indices.items()]
lp = (df['likpf_scale_5'].values.astype(float) - last) if 'likpf_scale_5' in df else None
g = df['well'].values
X = np.empty((len(df), len(feats)), dtype=np.float32)
for i, c in enumerate(feats):
    X[:, i] = df[c].values.astype(np.float32)
del df
gc.collect()
print(f'arrays ready, X={X.shape} ({time.time()-t0:.0f}s)', flush=True)
folds = list(GroupKFold(5).split(X, y, g))


def pooled_cv(meta_oof, w_sub1=0.60, tau=85.0, sg_win=61):
    """Stock post-process, so numbers are comparable with the kernel's own CV."""
    warm = 1.0 - np.exp(-np.maximum(mds, 0.0) / tau)
    delta = w_sub1 * warm * meta_oof + (1 - w_sub1) * (lp if lp is not None else 0.0)
    pred = last + delta
    out = pred.copy()
    for p in pos:
        v = pred[p]; wl = min(sg_win, len(v)); wl -= (wl % 2 == 0)
        if wl >= 5:
            out[p] = savgol_filter(v, wl, 3)
    return float(np.sqrt(np.mean((out - y_tvt) ** 2)))


def run_stack(models):
    oof_cols = {}
    for name, make, is_lgb in models:
        oof = np.zeros(len(y))
        for tr, va in folds:
            m = make()
            if is_lgb:
                m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                      eval_metric='rmse',
                      callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(0)])
                oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
            else:
                m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                      early_stopping_rounds=250, use_best_model=True, verbose=False)
                oof[va] = m.predict(X[va])
        oof_cols[name] = oof
        print(f'    {name}: raw OOF {np.sqrt(np.mean((y-oof)**2)):.4f}  '
              f'pooled {pooled_cv(oof):.4f}  ({time.time()-t0:.0f}s)', flush=True)
    OOF = pd.DataFrame(oof_cols)
    rid = Ridge(alpha=1.66, positive=True, fit_intercept=True)
    meta = np.zeros(len(y))
    for tr, va in folds:
        rid.fit(OOF.values[tr], y[tr]); meta[va] = rid.predict(OOF.values[va])
    return meta


LGB_BASE = dict(boosting_type='gbdt', objective='regression', verbose=-1,
                n_jobs=-1, max_bin=255, n_estimators=5000)

CONFIGS = {
    'baseline': [
        ('lgb0', lambda: lgb.LGBMRegressor(**LGB_BASE, num_leaves=255, min_child_samples=15,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0,
              reg_alpha=0.05, learning_rate=0.03, seed=123), True),
        ('lgb1', lambda: lgb.LGBMRegressor(**LGB_BASE, num_leaves=64, min_child_samples=40,
              subsample=0.474, subsample_freq=1, colsample_bytree=0.393, reg_lambda=95.75,
              reg_alpha=10.79, min_child_weight=0.24, learning_rate=0.0093, random_state=0), True),
    ],
    'deeper': [
        ('lgbD', lambda: lgb.LGBMRegressor(**LGB_BASE, num_leaves=511, min_child_samples=60,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=20.0,
              reg_alpha=1.0, learning_rate=0.02, seed=7), True),
    ],
    'huber': [
        ('lgbH', lambda: lgb.LGBMRegressor(**{**LGB_BASE, 'objective': 'huber'},
              num_leaves=255, min_child_samples=25, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=10.0, learning_rate=0.03, seed=11), True),
    ],
    'strongreg': [
        ('lgbR', lambda: lgb.LGBMRegressor(**LGB_BASE, num_leaves=127, min_child_samples=200,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=200.0,
              reg_alpha=20.0, learning_rate=0.02, seed=5), True),
    ],
}

for name, models in CONFIGS.items():
    if WANT and name not in WANT:
        continue
    print(f'\n=== {name} ===', flush=True)
    meta = run_stack(models)
    print(f'  STACK pooled CV = {pooled_cv(meta):.4f}', flush=True)
