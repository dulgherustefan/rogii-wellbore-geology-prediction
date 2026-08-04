"""Retrain the pipeline-B stack with the conservative-gs leg as a FEATURE.

The model has only ever seen the shipped-gs lik-PF (likpf_mean_d, its
highest-gain feature). Swapping the post-process leg to a gs=45 filter bought
-0.56 pooled CV on both well halves, which says that view carries signal the
model cannot currently reach -- it only enters through a fixed 0.40 weight.

Same architecture as the shipped stack (3 LGBM + 2 CatBoost -> positive Ridge,
GroupKFold by well, refit-on-all with the median best iteration) so the only
change is the added feature. Exports native model files + manifest.json, matching
what the inference kernel loads.

train_df_B.pkl / leg_train.pkl are our own dumps, so unpickling is safe.
"""
import numpy as np, pandas as pd, json, time, gc, warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.signal import savgol_filter
from proj import robfit
from pathlib import Path

OUT = Path('models_B14'); OUT.mkdir(exist_ok=True)
t0 = time.time()

df = pd.read_pickle('train_df_B.pkl')
feats = json.load(open('feats_B.json'))
leg = pd.read_pickle('leg_train.pkl')
df = df.merge(leg, on='id', how='left')
df['likpf_leg'] = df['likpf_leg'].fillna(df['likpf_scale_5'])
df['likpf_leg_d'] = (df['likpf_leg'] - df['last_known_tvt']).astype(np.float32)
# The four likelihood-temperature trajectories are computed on EVERY run and then
# thrown away: main_B's feature filter drops anything starting with "likpf_scale_".
# They cost nothing extra at inference (add_likpf_features already emits the _d
# columns) and give the model four more views of the same particle filter at
# different trust levels. Model gains have been the best-transferring changes
# measured (iter11: CV -0.086 -> LB -0.202), so this is the cheapest real lever left.
EXTRA = ['likpf_leg_d', 'likpf_scale_3_d', 'likpf_scale_5_d',
         'likpf_scale_8_d', 'likpf_scale_12_d']
feats = list(feats) + [c for c in EXTRA if c in df.columns]
print(f'{df.shape} feats={len(feats)} ({time.time()-t0:.0f}s)', flush=True)

y = df['target'].values.astype(np.float32)
g = df['well'].values
last = df['last_known_tvt'].values.astype(float)
y_tvt = last + y
mds = df['md_since'].values.astype(float)
zc = df['z'].values.astype(float)
legv = df['likpf_leg'].values.astype(float)
pos = [np.asarray(ix) for ix in df.groupby('well', sort=False).indices.values()]
rng = np.random.RandomState(0)
perm = rng.permutation(pd.unique(g))
mask = {'A': np.isin(g, perm[:len(perm) // 2]), 'B': np.isin(g, perm[len(perm) // 2:])}
X = np.empty((len(df), len(feats)), dtype=np.float32)
for i, c in enumerate(feats):
    X[:, i] = df[c].values.astype(np.float32)
del df
gc.collect()
print(f'arrays ready {X.shape} ({time.time()-t0:.0f}s)', flush=True)

folds = list(GroupKFold(5).split(X, y, g))
LGB = dict(boosting_type='gbdt', objective='regression', verbose=-1, n_jobs=-1, max_bin=255)
CFGS = [
    ('lgb0', lambda: LGBMRegressor(**LGB, num_leaves=255, min_child_samples=15,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0,
        reg_alpha=0.05, learning_rate=0.03, n_estimators=5000, seed=123), True),
    ('lgb1', lambda: LGBMRegressor(**LGB, num_leaves=64, min_child_samples=40,
        subsample=0.474, subsample_freq=1, colsample_bytree=0.393, reg_lambda=95.75,
        reg_alpha=10.79, min_child_weight=0.24, learning_rate=0.0093,
        n_estimators=10000, random_state=0), True),
    ('lgb2', lambda: LGBMRegressor(**LGB, num_leaves=64, min_child_samples=40,
        subsample=0.474, subsample_freq=1, colsample_bytree=0.393, reg_lambda=95.75,
        reg_alpha=10.79, min_child_weight=0.24, learning_rate=0.0093,
        n_estimators=10000, random_state=29), True),
    ('cb0', lambda: CatBoostRegressor(iterations=8000, depth=7, l2_leaf_reg=2.0,
        min_data_in_leaf=15, border_count=254, loss_function='RMSE', od_type='Iter',
        od_wait=300, verbose=0, learning_rate=0.02, random_seed=7), False),
    ('cb1', lambda: CatBoostRegressor(iterations=8000, depth=7, l2_leaf_reg=2.0,
        min_data_in_leaf=15, border_count=254, loss_function='RMSE', od_type='Iter',
        od_wait=300, verbose=0, learning_rate=0.03, random_seed=123), False),
]

oof_cols, fitted, order = {}, {}, []
for name, make, is_lgb in CFGS:
    oof = np.zeros(len(y)); iters = []
    for k, (tr, va) in enumerate(folds):
        m = make()
        if is_lgb:
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric='rmse',
                  callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])
            iters.append(int(m.best_iteration_ or 0))
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
        else:
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], early_stopping_rounds=250,
                  use_best_model=True, verbose=False)
            iters.append(int(m.get_best_iteration() or 0))
            oof[va] = m.predict(X[va])
        print(f'  {name} fold{k} ({time.time()-t0:.0f}s)', flush=True)
    oof_cols[name] = oof; order.append(name)
    print(f'{name}: OOF RMSE={np.sqrt(np.mean((y-oof)**2)):.4f} ({time.time()-t0:.0f}s)', flush=True)
    good = [i for i in iters if i > 0]
    mf = make()
    if good:
        mf.set_params(**({'n_estimators': max(int(np.median(good)), 50)} if is_lgb
                         else {'iterations': max(int(np.median(good)), 50)}))
    mf.fit(X, y) if is_lgb else mf.fit(X, y, verbose=False)
    fitted[name] = mf
    print(f'{name}: refit-on-all done ({time.time()-t0:.0f}s)', flush=True)

OOFM = np.column_stack([oof_cols[n] for n in order])
rid = Ridge(alpha=1.66, positive=True, fit_intercept=True)
meta = np.zeros(len(y))
for tr, va in folds:
    rid.fit(OOFM[tr], y[tr]); meta[va] = rid.predict(OOFM[va])
rid.fit(OOFM, y)
print(f'ridge-stack OOF RMSE={np.sqrt(np.mean((y-meta)**2)):.4f}', flush=True)

warm = 1 - np.exp(-np.maximum(mds, 0) / 85.0)
p = last + 0.65 * warm * meta + 0.35 * (legv - last)
o = p.copy()
for q in pos:
    v = p[q]; wl = min(61, len(v)); wl -= (wl % 2 == 0)
    if wl >= 5:
        v = savgol_filter(v, wl, 3)
    s = mds[q] / max(mds[q].max(), 1e-6)
    f = robfit(s, v + zc[q], 4) - zc[q]
    o[q] = 0.25 * v + 0.75 * f if np.all(np.isfinite(f)) else v
for k in ('A', 'B'):
    print(f'  half {k}: {np.sqrt(np.mean((o[mask[k]]-y_tvt[mask[k]])**2)):.4f}', flush=True)
print('  (iter11 reference: A=8.1982 B=8.5337 at w=0.65)')

for name in order:
    m = fitted[name]
    if name.startswith('lgb'):
        m.booster_.save_model(str(OUT / f'{name}.txt'))
    else:
        m.save_model(str(OUT / f'{name}.cbm'))
json.dump({'feats': feats, 'col_order': order,
           'ridge_coef': [float(x) for x in np.ravel(rid.coef_)],
           'ridge_intercept': float(rid.intercept_)}, open(OUT / 'manifest.json', 'w'))
json.dump(feats, open(OUT / 'features.json', 'w'))
np.save('meta_oof_b14.npy', meta)
print(f'wrote {OUT} ({time.time()-t0:.0f}s)')
