"""Experiment: can CatBoost + engineered features + LGBM blend beat OOF 10.62?
No feature regen — uses cached features.pkl. GroupKFold(5) by well."""
import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

t0 = time.time()
df = pd.read_pickle('features.pkl')
print(f"loaded {len(df)} rows, {df['wid'].nunique()} wells  ({time.time()-t0:.0f}s)")

BASE = ['pf_dev', 'pf2_dev', 'st_dev', 'disagree', 'disagree2',
        'st_kres', 'pf_cv', 'st_disp', 'cov',
        'md_into', 'md_frac', 'Z_dev', 'incl', 'lat_dist',
        'gr', 'grs', 'dgr', 'gr_dev_anchor', 'gr_rs']

# --- engineered consensus / confidence-interaction features (free, from cached cols) ---
def engineer(d):
    e = pd.DataFrame(index=d.index)
    e['consensus'] = (d['pf_dev'] + d['st_dev'] + d['pf2_dev']) / 3.0
    e['pf_st_mid'] = 0.5 * (d['pf_dev'] + d['st_dev'])
    e['abs_dis'] = d['disagree'].abs()
    e['abs_dis2'] = d['disagree2'].abs()
    # trust-weighted: struct trusted when known-residual low; pf when pseudo-cv low
    e['st_trust'] = d['st_dev'] / (1.0 + d['st_kres'])
    e['pf_trust'] = d['pf_dev'] / (1.0 + d['pf_cv'])
    e['pf_far'] = d['pf_dev'] * d['md_frac']      # extrapolation distance interaction
    e['st_far'] = d['st_dev'] * d['md_frac']
    e['dis_cov'] = d['disagree'] * np.log1p(d['cov'])
    e['fault'] = d['st_disp'] * d['md_frac']      # fault-risk grows with distance
    e['gr_mag'] = d['gr_dev_anchor'].abs()
    return e

eng = engineer(df)
ENG = list(eng.columns)
df = pd.concat([df, eng], axis=1)
FEATS = BASE + ENG

X = df[FEATS].values; y = df['y'].values; g = df['wid'].values
print(f"features: {len(FEATS)} ({len(BASE)} base + {len(ENG)} eng)")
print(f"const  RMSE={np.sqrt(np.mean(y**2)):.3f}  "
      f"PF={np.sqrt(np.mean((y-df['pf_dev'])**2)):.3f}  "
      f"struct={np.sqrt(np.mean((y-df['st_dev'])**2)):.3f}")

gkf = GroupKFold(5)
folds = list(gkf.split(X, y, g))

lgb_params = dict(n_estimators=1600, learning_rate=0.025, num_leaves=127,
                  min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1,
                  objective='regression', verbose=-1, n_jobs=-1)

oof_lgb = np.zeros(len(y)); oof_cat = np.zeros(len(y))
Xbase = df[BASE].values
for k, (tr, va) in enumerate(folds):
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[va] = m.predict(X[va])

    c = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=8,
                          l2_leaf_reg=6.0, loss_function='RMSE',
                          random_seed=42, thread_count=-1, verbose=0,
                          early_stopping_rounds=120)
    c.fit(Xbase[tr], y[tr], eval_set=(Xbase[va], y[va]), use_best_model=True)
    oof_cat[va] = c.predict(Xbase[va])
    print(f"fold{k}: lgb={np.sqrt(np.mean((y[va]-oof_lgb[va])**2)):.3f} "
          f"cat={np.sqrt(np.mean((y[va]-oof_cat[va])**2)):.3f}  ({time.time()-t0:.0f}s)")

def rmse(p): return np.sqrt(np.mean((y-p)**2))
print(f"\nLGBM (eng feats) OOF = {rmse(oof_lgb):.3f}")
print(f"CatBoost (base)  OOF = {rmse(oof_cat):.3f}")
# blend search
best = (1.0, rmse(oof_lgb))
for a in np.linspace(0, 1, 21):
    r = rmse(a*oof_lgb + (1-a)*oof_cat)
    if r < best[1]: best = (a, r)
print(f"best blend: {best[0]:.2f}*lgb + {1-best[0]:.2f}*cat -> OOF = {best[1]:.3f}")
print(f"(prev baseline OOF was 10.62)")
np.save('oof_lgb_eng.npy', oof_lgb); np.save('oof_cat.npy', oof_cat)
print(f"done {time.time()-t0:.0f}s")
