"""Does the slide-9 NCC signal lower OOF? Compare base vs base+NCC for
LGBM and CatBoost, then blend. GroupKFold(5) by well. No engineered feats
(those hurt: 10.79 vs 10.62)."""
import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

t0 = time.time()
df = pd.read_pickle('features3.pkl')
print(f"loaded {len(df)} rows, {df['wid'].nunique()} wells  ({time.time()-t0:.0f}s)")

BASE = ['pf_dev', 'pf2_dev', 'st_dev', 'disagree', 'disagree2',
        'st_kres', 'pf_cv', 'st_disp', 'cov',
        'md_into', 'md_frac', 'Z_dev', 'incl', 'lat_dist',
        'gr', 'grs', 'dgr', 'gr_dev_anchor', 'gr_rs']
NCC = ['ncc_cons_dev', 'ncc_sc', 'ncc8_dev', 'ncc8_sc',
       'ncc25_dev', 'ncc25_sc', 'ncc_dis']
print("NCC cols present:", [c for c in NCC if c in df.columns])

y = df['y'].values; g = df['wid'].values
folds = list(GroupKFold(5).split(df, y, g))
lgb_params = dict(n_estimators=1400, learning_rate=0.03, num_leaves=127,
                  min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1,
                  objective='regression', verbose=-1, n_jobs=-1)

def run_lgb(feats):
    X = df[feats].values; oof = np.zeros(len(y))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**lgb_params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va])
    return oof

def run_cat(feats):
    X = df[feats].values; oof = np.zeros(len(y))
    for tr, va in folds:
        c = CatBoostRegressor(iterations=2500, learning_rate=0.03, depth=8,
                              l2_leaf_reg=6.0, loss_function='RMSE', random_seed=42,
                              thread_count=-1, verbose=0, early_stopping_rounds=150)
        c.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True)
        oof[va] = c.predict(X[va])
    return oof

def rmse(p): return np.sqrt(np.mean((y-p)**2))

res = {}
for name, feats in [('lgb_base', BASE), ('lgb_ncc', BASE+NCC)]:
    res[name] = run_lgb(feats); print(f"{name:10s} OOF={rmse(res[name]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
for name, feats in [('cat_base', BASE), ('cat_ncc', BASE+NCC)]:
    res[name] = run_cat(feats); print(f"{name:10s} OOF={rmse(res[name]):.3f}  ({time.time()-t0:.0f}s)", flush=True)

# best single
best_lgb = min(['lgb_base', 'lgb_ncc'], key=lambda k: rmse(res[k]))
best_cat = min(['cat_base', 'cat_ncc'], key=lambda k: rmse(res[k]))
print(f"\nbest lgb={best_lgb} {rmse(res[best_lgb]):.3f} | best cat={best_cat} {rmse(res[best_cat]):.3f}")
L, C = res[best_lgb], res[best_cat]
best = (1.0, rmse(L))
for a in np.linspace(0, 1, 41):
    r = rmse(a*L + (1-a)*C)
    if r < best[1]: best = (a, r)
print(f"BEST BLEND {best[0]:.3f}*{best_lgb} + {1-best[0]:.3f}*{best_cat} -> OOF={best[1]:.3f}")
print(f"(baseline LGBM base no-NCC OOF was 10.62; LB 10.147)")
for k in res: np.save(f'oof_{k}.npy', res[k])
print(f"done {time.time()-t0:.0f}s")
