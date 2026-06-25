"""Does pf3 (responsive PF) lower OOF below the v3 blend (10.421)?
Compare v3-feats vs v3-feats+pf3 for LGBM & CatBoost, then blend. GroupKFold(5)."""
import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

t0 = time.time()
df = pd.read_pickle('features4.pkl')
print(f"loaded {len(df)} rows  ({time.time()-t0:.0f}s)")

BASE = ['pf_dev', 'pf2_dev', 'st_dev', 'disagree', 'disagree2',
        'st_kres', 'pf_cv', 'st_disp', 'cov',
        'md_into', 'md_frac', 'Z_dev', 'incl', 'lat_dist',
        'gr', 'grs', 'dgr', 'gr_dev_anchor', 'gr_rs']
NCC = ['ncc_cons_dev', 'ncc_sc', 'ncc8_dev', 'ncc8_sc', 'ncc25_dev', 'ncc25_sc', 'ncc_dis']
PF3 = ['pf3_dev', 'disagree3']
V3 = BASE + NCC          # the submitted-#1 feature set
V4 = BASE + NCC + PF3

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
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va])
    return oof

def run_cat(feats):
    X = df[feats].values; oof = np.zeros(len(y))
    for tr, va in folds:
        c = CatBoostRegressor(iterations=2500, learning_rate=0.03, depth=8, l2_leaf_reg=6.0,
                              loss_function='RMSE', random_seed=42, thread_count=-1,
                              verbose=0, early_stopping_rounds=150)
        c.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True); oof[va] = c.predict(X[va])
    return oof

def rmse(p): return np.sqrt(np.mean((y-p)**2))

res = {}
for name, feats in [('lgb_v3', V3), ('lgb_v4', V4)]:
    res[name] = run_lgb(feats); print(f"{name:8s} OOF={rmse(res[name]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
for name, feats in [('cat_v3', V3), ('cat_v4', V4)]:
    res[name] = run_cat(feats); print(f"{name:8s} OOF={rmse(res[name]):.3f}  ({time.time()-t0:.0f}s)", flush=True)

for tag, L, C in [('v3', res['lgb_v3'], res['cat_v3']), ('v4', res['lgb_v4'], res['cat_v4'])]:
    best = (1.0, rmse(L))
    for a in np.linspace(0, 1, 41):
        r = rmse(a*L+(1-a)*C)
        if r < best[1]: best = (a, r)
    print(f"{tag} BLEND {best[0]:.3f}*lgb+{1-best[0]:.3f}*cat -> OOF={best[1]:.3f}")
print("(v3 blend was 10.421; submit #2 only if v4 < that)")
print(f"done {time.time()-t0:.0f}s")
