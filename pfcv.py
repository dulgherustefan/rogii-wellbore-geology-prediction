"""Does a retuned PF actually move the pooled CV, not just the leg RMSE?

The post-process is  delta = 0.60*warm*meta + 0.40*(likpf_scale_5 - last), so a
better physics leg should show up directly. Leg RMSE alone is not enough evidence:
the leg's error is partly correlated with the learned model's, and the observed
seed-noise ratio was only ~0.17 CV per unit of leg RMSE.

Scores the SAME wells with the old and the retuned leg, holding meta_oof fixed,
through the full iter7 post-process (savgol + projection deg4/lam0.75).

Note the retuned PF is used ONLY for the post-process leg. The GBM's
likpf_scale_* features stay on the old dynamics, because the models were trained
on those and swapping them at inference would be a train/test mismatch.

Our own dumps, so unpickling is safe.
"""
import numpy as np, pandas as pd, sys, time, json
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from proj import robfit
import pfsweep as S

N_WELLS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OVER = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 96
OFFSET = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # disjoint well slice

t0 = time.time()
d = pd.read_pickle('oof_B.pkl').merge(
    pd.read_pickle('train_df_B.pkl')[['id', 'z']], on='id', how='left')
rng = np.random.RandomState(7)
wells = list(rng.permutation(d.well.unique())[OFFSET:OFFSET + N_WELLS])
d = d[d.well.isin(wells)].reset_index(drop=True)
print(f'{d.shape} wells={d.well.nunique()} override={OVER} seeds={SEEDS}', flush=True)

last = d.last_known_tvt.to_numpy(float)
y = last + d.target.to_numpy(float)
mds = d.md_since.to_numpy(float)
meta = d.meta_oof.to_numpy(float)
zc = d.z.to_numpy(float)
lp_old = d.likpf_scale_5.to_numpy(float)
gi = d.groupby('well', sort=False).indices
pos = [np.asarray(ix) for ix in gi.values()]


def pooled(lp):
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
    return float(np.sqrt(np.mean((o - y) ** 2)))


def leg_abs(wid, P):
    """Absolute TVT trajectory (leg_for_well returns pred-minus-truth)."""
    hw = pd.read_csv(f'data/train/{wid}__horizontal_well.csv')
    e = S.leg_for_well(wid, P, SEEDS, 350, 5.0)
    if e is None:
        return None
    t = hw['TVT_input'].to_numpy(float)
    ev = np.flatnonzero(~np.isfinite(t))
    return wid, e + hw['TVT'].to_numpy(float)[ev], ev


P = dict(S.DEF); P.update(OVER)
res = Parallel(n_jobs=6)(delayed(leg_abs)(w, P) for w in wells)
new = {}
for r in res:
    if r is None:
        continue
    wid, arr, ev = r
    new[wid] = dict(zip([f'{wid}_{i}' for i in ev], arr))
lp_new = np.array([new.get(w, {}).get(i, np.nan)
                   for w, i in zip(d.well.to_numpy(), d.id.to_numpy())])
miss = np.isnan(lp_new).sum()
lp_new = np.where(np.isnan(lp_new), lp_old, lp_new)
print(f'rebuilt leg for {len(new)} wells, {miss} rows fell back to old '
      f'({time.time()-t0:.0f}s)', flush=True)
print(f'  leg RMSE  old={np.sqrt(np.mean((lp_old-y)**2)):.4f}  '
      f'new={np.sqrt(np.mean((lp_new-y)**2)):.4f}')
print(f'  pooled CV old={pooled(lp_old):.4f}  new={pooled(lp_new):.4f}')
