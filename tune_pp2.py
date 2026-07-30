"""Exact post-process tuning by closed-form least squares.

Coordinate descent over the blend weights is slow and path-dependent. But the
prediction is LINEAR in those weights and the Savitzky-Golay smoother is a linear
operator, so smoothing each component once lets the optimal weights be solved
directly:

    pred = smooth(last) + c1*smooth(A1) + c2*smooth(A2) + ...

Only tau and the smoothing window change the basis, so those are the only things
that need a grid; everything else is exact. Weights are fitted with GroupKFold
over wells so the reported CV stays honest.

oof_B.pkl is written by our own training run, so unpickling it is safe.
"""
import numpy as np, pandas as pd, json
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold

df = pd.read_pickle('oof_B.pkl').reset_index(drop=True)
print(f'rows={len(df)} wells={df.well.nunique()}')

last = df['last_known_tvt'].to_numpy(float)
y = last + df['target'].to_numpy(float)
meta = df['meta_oof'].to_numpy(float)
mds = df['md_since'].to_numpy(float)
groups_id = df['well'].to_numpy()
pos_by_well = [np.asarray(ix) for _, ix in df.groupby('well', sort=False).indices.items()]

# surf_dev is the mean of the per-scale columns, so keeping both makes the
# design exactly collinear and the solved weights explode; drop the aggregate.
SURFS = [c for c in df.columns if c.startswith('surf_dev')]
if len(SURFS) > 1 and 'surf_dev' in SURFS:
    SURFS = [c for c in SURFS if c != 'surf_dev']
LPS = [c for c in df.columns if c.startswith('likpf_scale_') and not c.endswith('_d')]
print('surf cols:', SURFS, '\nlikpf cols:', LPS)


def smooth(v, win, poly=3):
    if win < 5:
        return v
    out = v.copy()
    for pos in pos_by_well:
        seg = v[pos]; n = len(seg); wl = min(win, n)
        if wl % 2 == 0:
            wl -= 1
        if wl >= poly + 2:
            out[pos] = savgol_filter(seg, wl, poly)
    return out


def basis(tau):
    warm = 1.0 - np.exp(-np.maximum(mds, 0.0) / tau) if tau > 1e-9 else np.ones_like(mds)
    cols = {'model': warm * meta}
    for c in LPS:
        cols[c] = df[c].to_numpy(float) - last
    for c in SURFS:
        cols[c] = df[c].to_numpy(float)
    return cols


RIDGE = 1e-2 * len(df)
folds = list(GroupKFold(5).split(np.zeros(len(df)), y, groups_id))


def evaluate(tau, win, keys):
    cols = basis(tau)
    B = np.column_stack([smooth(cols[k], win) for k in keys])
    base = smooth(last.copy(), win)
    tgt = y - base
    oof = np.zeros(len(y))
    for tr, va in folds:
        A = B[tr]
        coef = np.linalg.solve(A.T @ A + RIDGE * np.eye(B.shape[1]), A.T @ tgt[tr])
        oof[va] = B[va] @ coef
    cv = float(np.sqrt(np.mean((base + oof - y) ** 2)))
    coef_full = np.linalg.solve(B.T @ B + RIDGE * np.eye(B.shape[1]), B.T @ tgt)
    return cv, coef_full


KEYSETS = {
    'model_only': ['model'],
    'model+lp5': ['model', 'likpf_scale_5'],
    'model+lp3': ['model', 'likpf_scale_3'],
    'model+alllp': ['model'] + LPS,
    'model+lp5+surf': ['model', 'likpf_scale_5'] + SURFS,
    'model+lp3+surf': ['model', 'likpf_scale_3'] + SURFS,
    'model+alllp+surf': ['model'] + LPS + SURFS,
}

best = None
for tau in (1e-9, 85.0, 150.0, 300.0):
    for win in (61, 401, 801, 1601):
        for name, keys in KEYSETS.items():
            keys = [k for k in keys if k in basis(tau)]
            cv, coef = evaluate(tau, win, keys)
            if best is None or cv < best[0]:
                best = (cv, tau, win, name, keys, coef)
                print(f'  NEW BEST  tau={tau:<6g} win={win:<5d} {name:18s} CV={cv:.4f}', flush=True)

cv, tau, win, name, keys, coef = best
print(f'\nBEST CV = {cv:.4f}   tau={tau} win={win} set={name}')
print('weights:', {k: round(float(c), 4) for k, c in zip(keys, coef)})
print(f'estimated LB = {cv - 0.81:.3f}')
json.dump({'tau': float(tau), 'win': int(win), 'keys': keys,
           'coef': [float(c) for c in coef], 'cv': cv}, open('pp_best2.json', 'w'), indent=1)
print('wrote pp_best2.json')
