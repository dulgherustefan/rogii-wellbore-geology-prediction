"""Is GBM seed noise worth averaging away?

The orthonormal-basis decomposition showed my U-coefficient errors are unbiased
noise (optimal rescale 1.000 on every component), so the only remaining lever is
averaging. PF-seed averaging is already shipped (iter9). This measures the other
noise source: the boosters themselves.

Scores single-seed vs seed-averaged OOF through the FULL iter7 post-process
(savgol + projection deg4/lam0.75), because that is the number that maps to the
leaderboard. Ship a multi-seed retrain only if the gain justifies the hours.

train_df_B.pkl is our own dump, so unpickling it is safe.
"""
import numpy as np, pandas as pd, json, time, gc, sys, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scipy.signal import savgol_filter
from proj import robfit

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
SEEDS = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['123', '777'])]
NEST = int(sys.argv[3]) if len(sys.argv) > 3 else 1500

t0 = time.time()
df = pd.read_pickle('train_df_B.pkl')
feats = json.load(open('feats_B.json'))
keep = set(pd.Series(df.well.unique()).sample(frac=FRAC, random_state=0))
df = df[df.well.isin(keep)].reset_index(drop=True)
print(f'{df.shape} wells={df.well.nunique()} seeds={SEEDS}', flush=True)

y = df.target.values.astype(np.float32)
last = df.last_known_tvt.values.astype(float)
y_tvt = last + y
mds = df.md_since.values.astype(float)
zc = df.z.values.astype(float)
lp = df.likpf_scale_5.values.astype(float)
g = df.well.values
pos = [np.asarray(ix) for ix in df.groupby('well', sort=False).indices.values()]
X = np.empty((len(df), len(feats)), dtype=np.float32)
for i, c in enumerate(feats):
    X[:, i] = df[c].values.astype(np.float32)
del df
gc.collect()
folds = list(GroupKFold(5).split(X, y, g))


def pooled(meta):
    warm = 1 - np.exp(-np.maximum(mds, 0) / 85.0)
    p = last + 0.60 * warm * meta + 0.40 * (lp - last)
    o = p.copy()
    for q in pos:
        v = p[q]; wl = min(61, len(v)); wl -= (wl % 2 == 0)
        if wl >= 5:
            v = savgol_filter(v, wl, 3)
        s = mds[q] / max(mds[q].max(), 1e-6)
        f = robfit(s, v + zc[q], 4) - zc[q]
        o[q] = 0.25 * v + 0.75 * f if np.all(np.isfinite(f)) else v
    return float(np.sqrt(np.mean((o - y_tvt) ** 2)))


BASE = dict(boosting_type='gbdt', objective='regression', verbose=-1, n_jobs=-1,
            max_bin=255, n_estimators=NEST)
oofs = []
for sd in SEEDS:
    oof = np.zeros(len(y))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**BASE, num_leaves=255, min_child_samples=15,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=3.0, reg_alpha=0.05, learning_rate=0.03,
                              seed=sd, random_state=sd)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric='rmse',
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
    oofs.append(oof)
    print(f'seed{sd}: single={pooled(oof):.4f}  avg_of_{len(oofs)}={pooled(np.mean(oofs, 0)):.4f}'
          f'  ({time.time()-t0:.0f}s)', flush=True)
