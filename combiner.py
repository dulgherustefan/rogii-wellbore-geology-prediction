"""Replace the fixed 0.60/0.40 blend with a learned combiner.

The post-process currently mixes the learned model and the physics leg with
constant weights and an exponential warm-up. That is crude: how much to trust the
leg plausibly depends on how far past the anchor we are, and on how much the two
legs disagree (disagreement is a label-free signal that the tracker is unsure).

Trained on OOF inputs only (meta_oof is already out-of-fold), with GroupKFold by
well, and reported on two disjoint halves of the wells. Post-process tuning has
burned this project once (tune_pp2: OOF -0.05, leaderboard +0.208), so a candidate
counts only if it wins on BOTH halves against the shipped recipe.

Our own dumps, so unpickling is safe.
"""
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scipy.signal import savgol_filter
from proj import robfit

t0 = time.time()
d = pd.read_pickle('oof_B.pkl').merge(
    pd.read_pickle('train_df_B.pkl')[['id', 'z']], on='id', how='left')
d = d.merge(pd.read_pickle('leg_train.pkl'), on='id', how='left')
d['likpf_leg'] = d['likpf_leg'].fillna(d['likpf_scale_5'])

last = d.last_known_tvt.to_numpy(float)
y_tvt = last + d.target.to_numpy(float)
tgt = d.target.to_numpy(np.float32)
mds = d.md_since.to_numpy(float)
meta = d.meta_oof.to_numpy(float)
zc = d.z.to_numpy(float)
wells = d.well.to_numpy()
pos = [np.asarray(ix) for ix in d.groupby('well', sort=False).indices.values()]
rng = np.random.RandomState(0)
perm = rng.permutation(pd.unique(wells))
mask = {'A': np.isin(wells, perm[:len(perm) // 2]),
        'B': np.isin(wells, perm[len(perm) // 2:])}

leg_new = d.likpf_leg.to_numpy(float) - last
leg_old = d.likpf_scale_5.to_numpy(float) - last
warm = 1 - np.exp(-np.maximum(mds, 0) / 85.0)

X = np.column_stack([
    meta, leg_new, leg_old, leg_new - leg_old,      # the two legs and their disagreement
    mds, warm, meta * warm,
    d.likpf_scale_3.to_numpy(float) - last,
    d.likpf_scale_12.to_numpy(float) - last,
    d.pf_ancc.to_numpy(float) - last,
]).astype(np.float32)
NAMES = ['meta', 'leg_new', 'leg_old', 'leg_disagree', 'md_since', 'warm',
         'meta_warm', 'pf3', 'pf12', 'pf_ancc']


def finish(delta):
    """Same projection post-process as iter7/iter10, applied to any delta."""
    p = last + delta
    o = p.copy()
    for q in pos:
        v = p[q]; wl = min(61, len(v)); wl -= (wl % 2 == 0)
        if wl >= 5:
            v = savgol_filter(v, wl, 3)
        s = mds[q] / max(mds[q].max(), 1e-6)
        f = robfit(s, v + zc[q], 4) - zc[q]
        o[q] = 0.25 * v + 0.75 * f if np.all(np.isfinite(f)) else v
    return o


def halves(pred):
    return tuple(float(np.sqrt(np.mean((pred[mask[k]] - y_tvt[mask[k]]) ** 2)))
                 for k in ('A', 'B'))


if __name__ == '__main__':
    a0, b0 = halves(finish(0.60 * warm * meta + 0.40 * leg_new))
    print(f'shipped linear blend (new leg): A={a0:.4f} B={b0:.4f}', flush=True)

    oof = np.zeros(len(tgt))
    for k, (tr, va) in enumerate(GroupKFold(5).split(X, tgt, wells)):
        m = lgb.LGBMRegressor(objective='regression', n_estimators=1200,
                              learning_rate=0.04, num_leaves=63,
                              min_child_samples=200, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.9,
                              reg_lambda=20.0, verbose=-1, n_jobs=-1, seed=1)
        m.fit(X[tr], tgt[tr], eval_set=[(X[va], tgt[va])], eval_metric='rmse',
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
        print(f'  fold{k} done ({time.time()-t0:.0f}s)', flush=True)
    a1, b1 = halves(finish(oof))
    print(f'learned combiner:              A={a1:.4f} B={b1:.4f}')
    for w in (0.3, 0.5, 0.7):
        a2, b2 = halves(finish(w * oof + (1 - w) * (0.60 * warm * meta + 0.40 * leg_new)))
        print(f'  {w:.1f}*combiner + {1-w:.1f}*linear:   A={a2:.4f} B={b2:.4f}')
    imp = sorted(zip(NAMES, m.feature_importances_), key=lambda t: -t[1])
    print('importance:', [f'{n}:{v}' for n, v in imp])
