"""Build the FINAL model bundle: LGBM + CatBoost ensemble on the winning feature
set, refit on ALL train wells (multi-seed), saved for the inference kernel.
Usage: python3 train_final.py <use_ncc:0|1> <blend_alpha>
bundle = {'lgb_models':[...], 'cat_models':[...], 'feats':FE, 'alpha':a, 'use_ncc':bool}
alpha = weight on LGBM; (1-alpha) on CatBoost."""
import sys, pandas as pd, numpy as np, warnings, time, joblib
warnings.filterwarnings('ignore')
import lightgbm as lgb
from catboost import CatBoostRegressor

USE_NCC = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
USE_PF3 = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

BASE = ['pf_dev', 'pf2_dev', 'st_dev', 'disagree', 'disagree2',
        'st_kres', 'pf_cv', 'st_disp', 'cov',
        'md_into', 'md_frac', 'Z_dev', 'incl', 'lat_dist',
        'gr', 'grs', 'dgr', 'gr_dev_anchor', 'gr_rs']
NCC = ['ncc_cons_dev', 'ncc_sc', 'ncc8_dev', 'ncc8_sc',
       'ncc25_dev', 'ncc25_sc', 'ncc_dis']
PF3 = ['pf3_dev', 'disagree3']
FE = BASE + (NCC if USE_NCC else []) + (PF3 if USE_PF3 else [])
PKL = 'features4.pkl' if USE_PF3 else ('features3.pkl' if USE_NCC else 'features.pkl')

t0 = time.time()
df = pd.read_pickle(PKL)
X = df[FE].values; y = df['y'].values
print(f"{PKL}: {len(df)} rows, {len(FE)} feats, use_ncc={USE_NCC}, alpha={ALPHA}")

lgb_params = dict(n_estimators=1200, learning_rate=0.03, num_leaves=127,
                  min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1,
                  objective='regression', verbose=-1, n_jobs=-1)
lgb_models = []
for s in (42, 7, 2024):
    p = dict(lgb_params); p['random_state'] = s
    m = lgb.LGBMRegressor(**p); m.fit(X, y); lgb_models.append(m)
    print(f"  lgb seed {s} done ({time.time()-t0:.0f}s)", flush=True)

cat_models = []
if ALPHA < 1.0:  # only train CatBoost if it's actually used in the blend
    for s in (42, 2024):
        c = CatBoostRegressor(iterations=1600, learning_rate=0.03, depth=8,
                              l2_leaf_reg=6.0, loss_function='RMSE', random_seed=s,
                              thread_count=-1, verbose=0)
        c.fit(X, y); cat_models.append(c)
        print(f"  cat seed {s} done ({time.time()-t0:.0f}s)", flush=True)

bundle = {'lgb_models': lgb_models, 'cat_models': cat_models,
          'feats': FE, 'alpha': ALPHA, 'use_ncc': USE_NCC}
joblib.dump(bundle, 'model_ds/rogii_gbm.joblib', compress=3)
print(f"saved model_ds/rogii_gbm.joblib ({time.time()-t0:.0f}s)")
