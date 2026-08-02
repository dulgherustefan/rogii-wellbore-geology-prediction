"""Does the prefix-backtest bias predict the FINAL pipeline's per-well level error?

vpcal.py measures the PF tracker's bias on a held-out tail of the visible prefix.
What actually has to be corrected is the level error of the blended submission,
so the correlation that matters is between that backtest bias and the per-well
mean error of the real OOF prediction -- not the PF's own bias.

Reports the achievable pooled CV for a shrunk correction pred -> pred - k*bias,
with k swept, since the optimal shrink is rho * sigma_eval / sigma_cal and
over-correcting is worse than not correcting at all.

Our own dumps, so unpickling is safe.
"""
import numpy as np, pandas as pd
from scipy.signal import savgol_filter

d = pd.read_pickle('oof_B.pkl')
cal = pd.read_csv('vpcal.csv')
CUTS = (0.50, 0.65, 0.75)

last = d['last_known_tvt'].to_numpy(float)
y = last + d['target'].to_numpy(float)
mds = d['md_since'].to_numpy(float)
meta = d['meta_oof'].to_numpy(float)
lp = d['likpf_scale_5'].to_numpy(float)
wells = d['well'].to_numpy()
groups = d.groupby('well', sort=False).indices
pos = {w: np.asarray(ix) for w, ix in groups.items()}

warm = 1 - np.exp(-np.maximum(mds, 0) / 85.0)
pred = last + 0.60 * warm * meta + 0.40 * (lp - last)
base = pred.copy()
for q in pos.values():
    v = pred[q]; wl = min(61, len(v)); wl -= (wl % 2 == 0)
    if wl >= 5:
        base[q] = savgol_filter(v, wl, 3)
err = base - y


def pooled(p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


lev = {w: float(err[q].mean()) for w, q in pos.items()}
cal = cal[cal['well'].isin(pos)].copy()
cal['lev_final'] = cal['well'].map(lev)
cal['bias_avg'] = cal[[f'bias_{f}' for f in CUTS]].mean(axis=1)

print(f'wells with both: {len(cal)}')
print(f'{"quantity":22s} {"rho vs PF bias_eval":>20s} {"rho vs FINAL level":>20s}')
for c in [f'bias_{f}' for f in CUTS] + ['bias_avg']:
    m = cal[c].notna()
    r1 = np.corrcoef(cal.loc[m, c], cal.loc[m, 'bias_eval'])[0, 1]
    r2 = np.corrcoef(cal.loc[m, c], cal.loc[m, 'lev_final'])[0, 1]
    print(f'{c:22s} {r1:+20.3f} {r2:+20.3f}')

print(f'\nbaseline pooled CV = {pooled(base):.4f}')
sd_c = cal['bias_avg'].std(); sd_l = cal['lev_final'].std()
print(f'sd(bias_avg)={sd_c:.2f}  sd(level)={sd_l:.2f}')
for k in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
    p = base.copy()
    for w, b in zip(cal['well'], cal['bias_avg']):
        if np.isfinite(b):
            p[pos[w]] -= k * b
    # score only on the wells that got a correction, so the number is not diluted
    mask = np.isin(wells, cal.loc[cal['bias_avg'].notna(), 'well'].to_numpy())
    print(f'  k={k:.1f}: pooled(corrected wells)={np.sqrt(np.mean((p[mask]-y[mask])**2)):.4f}'
          f'   [uncorrected on same wells={np.sqrt(np.mean((base[mask]-y[mask])**2)):.4f}]')
