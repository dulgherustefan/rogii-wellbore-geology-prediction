"""Train the pipeline-B stack on the dumped table and persist the OOF.

The post-process already blends a physics leg at weight 0.40 (raw
likpf_scale_5). Testing better physics legs -- the routed SP45 selector, the
beam ensemble, hold-shrinkage -- needs the learned leg's OOF held fixed, so it
is worth paying for it once and caching.

train_df_B.pkl is written by our own training run, so unpickling it is safe.
"""
import numpy as np, pandas as pd, json, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

t0 = time.time()
df = pd.read_pickle('train_df_B.pkl')
feats = json.load(open('feats_B.json'))
print(f'loaded {df.shape} feats={len(feats)}', flush=True)

y = df['target'].values.astype(np.float32)
ids = df['id'].values
g = df['well'].values
X = np.empty((len(df), len(feats)), dtype=np.float32)
for i, c in enumerate(feats):
    X[:, i] = df[c].values.astype(np.float32)
del df
gc.collect()
print(f'arrays ready {X.shape} ({time.time()-t0:.0f}s)', flush=True)

folds = list(GroupKFold(5).split(X, y, g))
LGB_BASE = dict(boosting_type='gbdt', objective='regression', verbose=-1,
                n_jobs=-1, max_bin=255, n_estimators=5000)
MODELS = [
    ('lgb0', lambda: lgb.LGBMRegressor(**LGB_BASE, num_leaves=255, min_child_samples=15,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0,
        reg_alpha=0.05, learning_rate=0.03, seed=123)),
    ('lgb1', lambda: lgb.LGBMRegressor(**LGB_BASE, num_leaves=64, min_child_samples=40,
        subsample=0.474, subsample_freq=1, colsample_bytree=0.393, reg_lambda=95.75,
        reg_alpha=10.79, min_child_weight=0.24, learning_rate=0.0093, random_state=0)),
]

cols = {}
for name, make in MODELS:
    oof = np.zeros(len(y))
    for k, (tr, va) in enumerate(folds):
        m = make()
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric='rmse',
              callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
        print(f'  {name} fold{k} ({time.time()-t0:.0f}s)', flush=True)
    cols[name] = oof
    print(f'{name}: raw OOF {np.sqrt(np.mean((y-oof)**2)):.4f}', flush=True)

OOF = np.column_stack([cols[n] for n, _ in MODELS])
rid = Ridge(alpha=1.66, positive=True, fit_intercept=True)
meta = np.zeros(len(y))
for tr, va in folds:
    rid.fit(OOF[tr], y[tr]); meta[va] = rid.predict(OOF[va])
print(f'meta raw OOF {np.sqrt(np.mean((y-meta)**2)):.4f}', flush=True)
pd.DataFrame({'id': ids, 'meta_oof': meta.astype(np.float32)}).to_pickle('meta_oof.pkl')
print(f'wrote meta_oof.pkl ({time.time()-t0:.0f}s)')
