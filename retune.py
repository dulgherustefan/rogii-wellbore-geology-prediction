"""Retune the blend against the NEW conservative-gs leg.

w_sub1 = 0.60 (learned model) / 0.40 (physics leg) was tuned when the leg was the
shipped-gs lik-PF. The gs=45 leg is a different estimator -- worse standalone but
decorrelated from the model -- so the optimal split has probably moved.

Post-process tuning has burned this project once already (tune_pp2 improved OOF
9.1921 -> 9.1398 and cost +0.208 on the leaderboard), so every candidate is scored
on TWO disjoint halves of the wells and only a setting that wins on both is
considered. A win on one half is treated as noise.

Our own dumps, so unpickling is safe.
"""
import numpy as np, pandas as pd, itertools, sys
from scipy.signal import savgol_filter
from proj import robfit

d = pd.read_pickle('oof_B.pkl').merge(
    pd.read_pickle('train_df_B.pkl')[['id', 'z']], on='id', how='left')
leg = pd.read_pickle('leg_train.pkl')
d = d.merge(leg, on='id', how='left')
miss = d.likpf_leg.isna().sum()
d['likpf_leg'] = d['likpf_leg'].fillna(d['likpf_scale_5'])
print(f'{d.shape} leg missing rows filled: {miss}', flush=True)

last = d.last_known_tvt.to_numpy(float)
y = last + d.target.to_numpy(float)
mds = d.md_since.to_numpy(float)
meta = d.meta_oof.to_numpy(float)
zc = d.z.to_numpy(float)
LEG = {'old': d.likpf_scale_5.to_numpy(float), 'new': d.likpf_leg.to_numpy(float)}

wells = d.well.to_numpy()
uw = pd.unique(wells)
rng = np.random.RandomState(0)
perm = rng.permutation(uw)
half = {'A': set(perm[:len(perm) // 2]), 'B': set(perm[len(perm) // 2:])}
mask = {k: np.isin(wells, list(v)) for k, v in half.items()}
groups = d.groupby('well', sort=False).indices
pos = [np.asarray(ix) for ix in groups.values()]


def score(legname, w_sub1, tau, deg, lam, sel, mix=None):
    # mix blends the two legs instead of choosing one: the conservative-gs leg is
    # decorrelated from the shipped one, so their average may beat either
    lp = LEG[legname] if mix is None else (1 - mix) * LEG['old'] + mix * LEG['new']
    warm = 1 - np.exp(-np.maximum(mds, 0) / tau)
    p = last + w_sub1 * warm * meta + (1 - w_sub1) * (lp - last)
    o = p.copy()
    for q in pos:
        v = p[q]; wl = min(61, len(v)); wl -= (wl % 2 == 0)
        if wl >= 5:
            v = savgol_filter(v, wl, 3)
        s = mds[q] / max(mds[q].max(), 1e-6)
        f = robfit(s, v + zc[q], deg) - zc[q]
        o[q] = (1 - lam) * v + lam * f if np.all(np.isfinite(f)) else v
    return float(np.sqrt(np.mean((o[sel] - y[sel]) ** 2)))


if __name__ == '__main__':
    print(f'{"config":42s} {"halfA":>8s} {"halfB":>8s} {"worst":>8s}')
    rows = []
    for legname in ('old', 'new'):
        for w in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            a = score(legname, w, 85.0, 4, 0.75, mask['A'])
            b = score(legname, w, 85.0, 4, 0.75, mask['B'])
            rows.append((legname, w, 85.0, 4, 0.75, a, b))
            print(f'leg={legname} w={w:.2f} tau=85 deg=4 lam=.75      '
                  f'{a:8.4f} {b:8.4f} {max(a,b):8.4f}', flush=True)
    for mix in (0.25, 0.5, 0.75):
        for w in (0.55, 0.60, 0.65):
            a = score('mix', w, 85.0, 4, 0.75, mask['A'], mix=mix)
            b = score('mix', w, 85.0, 4, 0.75, mask['B'], mix=mix)
            rows.append((f'mix{mix}', w, 85.0, 4, 0.75, a, b))
            print(f'leg=mix{mix} w={w:.2f} tau=85 deg=4 lam=.75     '
                  f'{a:8.4f} {b:8.4f} {max(a,b):8.4f}', flush=True)
    base = [r for r in rows if r[0] == 'old' and r[1] == 0.60][0]
    print(f'\nshipped baseline (old leg, w=0.60): A={base[5]:.4f} B={base[6]:.4f}')
    best = min(rows, key=lambda r: max(r[5], r[6]))
    print(f'best by WORST half: leg={best[0]} w={best[1]} -> A={best[5]:.4f} B={best[6]:.4f}')
