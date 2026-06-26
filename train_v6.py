"""Validate (GroupKFold OOF) and build the v6 model: v5 feats (base+pf3) +
plane-fit struct + GR-offsets. LGBM-only (CatBoost overfit before). Saves bundle."""
import sys, pandas as pd, numpy as np, warnings, time, joblib
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

t0 = time.time()
df = pd.read_pickle('features5.pkl')
print(f"loaded {len(df)} rows, {df['wid'].nunique()} wells ({time.time()-t0:.0f}s)")

BASE = ['pf_dev', 'pf2_dev', 'st_dev', 'disagree', 'disagree2',
        'st_kres', 'pf_cv', 'st_disp', 'cov',
        'md_into', 'md_frac', 'Z_dev', 'incl', 'lat_dist',
        'gr', 'grs', 'dgr', 'gr_dev_anchor', 'gr_rs']
PF3 = ['pf3_dev', 'disagree3']
NEW = ['stp_dev', 'stp_kres', 'stp_dis', 'stp_pf_dis',
       'tda_m20', 'tda_m8', 'tda_0', 'tda_p8', 'tda_p20']
V5 = BASE + PF3
V6 = BASE + PF3 + NEW
print("new cols present:", [c for c in NEW if c in df.columns])

y = df['y'].values; g = df['wid'].values
folds = list(GroupKFold(5).split(df, y, g))
params = dict(n_estimators=1400, learning_rate=0.03, num_leaves=127,
              min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1,
              objective='regression', verbose=-1, n_jobs=-1)

def oof(feats):
    X = df[feats].values; o = np.zeros(len(y))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        o[va] = m.predict(X[va])
    return np.sqrt(np.mean((y-o)**2))

if len(sys.argv) > 1 and sys.argv[1] == 'oof':
    print(f"V5 (base+pf3)         OOF={oof(V5):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"V6 (+plane+offsets)   OOF={oof(V6):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print("(v5 LB was 10.130; known-zone plane RMSE < IDW so plane is a real surface gain)")
    sys.exit(0)

# build final v6 model on all data (3 seeds, LGBM-only, alpha=1.0)
X = df[V6].values
models = []
for s in (42, 7, 2024):
    p = dict(params); p['n_estimators'] = 1000; p['random_state'] = s
    m = lgb.LGBMRegressor(**p); m.fit(X, y); models.append(m)
    print(f"  lgb seed {s} done ({time.time()-t0:.0f}s)", flush=True)
bundle = {'lgb_models': models, 'cat_models': [], 'feats': V6, 'alpha': 1.0, 'use_ncc': True}
joblib.dump(bundle, 'model_ds/rogii_gbm.joblib', compress=3)
print(f"saved v6 bundle ({len(V6)} feats) ({time.time()-t0:.0f}s)")
