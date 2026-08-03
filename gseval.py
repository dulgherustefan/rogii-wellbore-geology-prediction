"""Score every conservative-gs leg on all 773 wells, split into two disjoint halves.

The gs=45 choice came from 200-well slices and the response looked spiky there
(neighbours 40 and 50 gave much less on holdout). With the leg precomputed for all
773 wells this can be settled properly: same wells, same learned OOF, only the leg
differs, and a value is only credible if it wins on BOTH halves.

Our own dumps, so unpickling is safe.
"""
import numpy as np, pandas as pd, glob, re, sys
from scipy.signal import savgol_filter
from proj import robfit

W_SUB1 = 0.60
d = pd.read_pickle('oof_B.pkl').merge(
    pd.read_pickle('train_df_B.pkl')[['id', 'z']], on='id', how='left')
last = d.last_known_tvt.to_numpy(float)
y = last + d.target.to_numpy(float)
mds = d.md_since.to_numpy(float)
meta = d.meta_oof.to_numpy(float)
zc = d.z.to_numpy(float)
ids = d.id.to_numpy()
wells = d.well.to_numpy()
pos = [np.asarray(ix) for ix in d.groupby('well', sort=False).indices.values()]

rng = np.random.RandomState(0)
perm = rng.permutation(pd.unique(wells))
mask = {'A': np.isin(wells, perm[:len(perm) // 2]),
        'B': np.isin(wells, perm[len(perm) // 2:])}
warm = 1 - np.exp(-np.maximum(mds, 0) / 85.0)


def score(lp):
    p = last + W_SUB1 * warm * meta + (1 - W_SUB1) * (lp - last)
    o = p.copy()
    for q in pos:
        v = p[q]; wl = min(61, len(v)); wl -= (wl % 2 == 0)
        if wl >= 5:
            v = savgol_filter(v, wl, 3)
        s = mds[q] / max(mds[q].max(), 1e-6)
        f = robfit(s, v + zc[q], 4) - zc[q]
        o[q] = 0.25 * v + 0.75 * f if np.all(np.isfinite(f)) else v
    return tuple(float(np.sqrt(np.mean((o[mask[k]] - y[mask[k]]) ** 2)))
                 for k in ('A', 'B'))


if __name__ == '__main__':
    cands = [('shipped gs=clip(10,60)', d.likpf_scale_5.to_numpy(float))]
    for f in sorted(glob.glob('leg_*.pkl')) + ['leg_train.pkl']:
        m = re.search(r'leg_(\d+)\.pkl', f)
        tag = f'gs={m.group(1)}' if m else 'gs=45'
        col = pd.read_pickle(f).set_index('id')['likpf_leg']
        lp = pd.Series(ids).map(col).to_numpy(float)
        n_na = int(np.isnan(lp).sum())
        lp = np.where(np.isnan(lp), d.likpf_scale_5.to_numpy(float), lp)
        cands.append((f'{tag} ({n_na} na)', lp))
    # averaging several conservative legs: the gs response was spiky, so an
    # average over gs should damp the per-well mode-flipping without needing to
    # identify a single best value
    legs_only = [lp for tag, lp in cands[1:]]
    if len(legs_only) >= 3:
        cands.append(('AVG of all gs legs', np.mean(legs_only, axis=0)))
        mid = [lp for tag, lp in cands[1:] if any(k in tag for k in ('40', '45', '50'))]
        if len(mid) >= 2:
            cands.append(('AVG gs 40/45/50', np.mean(mid, axis=0)))
    print(f'{"leg":28s} {"halfA":>8s} {"halfB":>8s} {"mean":>8s}')
    rows = []
    for tag, lp in cands:
        a, b = score(lp)
        rows.append((tag, a, b))
        print(f'{tag:28s} {a:8.4f} {b:8.4f} {(a+b)/2:8.4f}', flush=True)
    base = rows[0]
    best = min(rows[1:], key=lambda r: max(r[1], r[2])) if len(rows) > 1 else None
    if best:
        print(f'\nbaseline {base[1]:.4f}/{base[2]:.4f}  ->  best {best[0]}: '
              f'{best[1]:.4f}/{best[2]:.4f}')
