"""Train GBM stacker on cached features. Target = TVT - anchor.
GroupKFold by well. Reports pooled RMSE vs PF and struct baselines."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

df = pd.read_pickle('features.pkl')
print(f"loaded {len(df)} rows, {df['wid'].nunique()} wells")

FEATS = ['pf_dev', 'pf2_dev', 'st_dev', 'disagree', 'disagree2',
         'st_kres', 'pf_cv', 'st_disp', 'cov',
         'md_into', 'md_frac', 'Z_dev', 'incl', 'lat_dist',
         'gr', 'grs', 'dgr', 'gr_dev_anchor', 'gr_rs']
X = df[FEATS].values; y = df['y'].values; g = df['wid'].values
w_n = df.groupby('wid')['y'].transform('size')  # for per-row, not used directly

def pooled_rmse(yt, yp, grp):
    # pool per well to match metric (each eval row counts once; already row-level)
    return np.sqrt(np.mean((yt-yp)**2))

# baselines (dev space; const = predict 0)
print(f"const  RMSE(dev)={np.sqrt(np.mean(y**2)):.3f}")
print(f"PF     RMSE(dev)={np.sqrt(np.mean((y-df['pf_dev'].values)**2)):.3f}")
print(f"struct RMSE(dev)={np.sqrt(np.mean((y-df['st_dev'].values)**2)):.3f}")

gkf = GroupKFold(5)
oof = np.zeros(len(y))
params = dict(n_estimators=1200, learning_rate=0.03, num_leaves=127,
              min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1,
              objective='regression', verbose=-1, n_jobs=-1)
imps = np.zeros(len(FEATS))
for tr, va in gkf.split(X, y, g):
    m = lgb.LGBMRegressor(**params)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(80, verbose=False)])
    oof[va] = m.predict(X[va]); imps += m.feature_importances_

print(f"\nGBM stacker OOF RMSE(dev)={np.sqrt(np.mean((y-oof)**2)):.3f}")
# clip predictions to sane range and re-eval
oc = np.clip(oof, -90, 90)
print(f"GBM stacker (clipped)   ={np.sqrt(np.mean((y-oc)**2)):.3f}")
print("\nfeature importance:")
for f, i in sorted(zip(FEATS, imps), key=lambda x: -x[1]):
    print(f"  {f:14s} {i:.0f}")
np.save('oof_stacker.npy', oof)
