"""Retune pipeline-B post-processing offline against the pooled well-level CV.

The post-process (warm-up damping, model/lik-PF blend, Savitzky-Golay smoothing)
still carries the original author's constants, tuned for a different feature set.
Because oof_B.pkl holds every column make_prediction_B consumes plus the
cross-fitted meta predictions, the whole thing can be re-optimised in seconds
instead of paying for a 5-hour retrain per trial.

Also tests promoting the structural surface from a mere feature to part of the
prediction backbone: delta += w_surf * surf_dev.

oof_B.pkl is written by our own training run, so unpickling it is safe.
"""
import numpy as np, pandas as pd, itertools, sys
from scipy.signal import savgol_filter

df = pd.read_pickle('oof_B.pkl')
print(f'rows={len(df)}  wells={df.well.nunique()}  cols={len(df.columns)}')

last = df['last_known_tvt'].to_numpy(float)
y_true = last + df['target'].to_numpy(float)
meta = df['meta_oof'].to_numpy(float)
mds = df['md_since'].to_numpy(float)
pf_delta = df['pf_ancc'].to_numpy(float) - last if 'pf_ancc' in df else np.zeros(len(df))
surf = df['surf_dev'].to_numpy(float) if 'surf_dev' in df else np.zeros(len(df))
LP = {c: df[c].to_numpy(float) - last for c in df.columns if c.startswith('likpf_scale_')}
print('lik-PF scales:', list(LP))

# group row positions once; the smoother runs per well
groups = [np.asarray(ix) for _, ix in df.reset_index(drop=True).groupby('well', sort=False).indices.items()]


def rmse(p):
    return float(np.sqrt(np.mean((p - y_true) ** 2)))


def predict(alpha, tau, w_pf, w_sub1, lp_key, w_surf, sg_win, sg_poly=3,
            trend_deg=0):
    warm = 1.0 - np.exp(-np.maximum(mds, 0.0) / tau) if tau > 1e-6 else 1.0
    sub1 = alpha * warm * (meta * (1 - w_pf) + pf_delta * w_pf)
    lp = LP.get(lp_key, pf_delta)
    delta = w_sub1 * sub1 + (1 - w_sub1) * lp + w_surf * surf
    pred = last + delta
    if trend_deg:
        # The recoverable signal is a low-order trend per well, so replacing the
        # local smoother with a global polynomial fit is a stronger regulariser.
        out = pred.copy()
        for pos in groups:
            v = pred[pos]
            if len(v) < trend_deg + 3:
                continue
            xx = np.linspace(-1.0, 1.0, len(v))
            out[pos] = np.polyval(np.polyfit(xx, v, trend_deg), xx)
        return out
    if sg_win < 5:
        return pred
    out = pred.copy()
    for pos in groups:
        v = pred[pos]; n = len(v); wl = min(sg_win, n)
        if wl % 2 == 0:
            wl -= 1
        if wl >= sg_poly + 2:
            out[pos] = savgol_filter(v, wl, sg_poly)
    return out


BASE = dict(alpha=1.0, tau=85.0, w_pf=0.0, w_sub1=0.60,
            lp_key='likpf_scale_5', w_surf=0.0, sg_win=61, trend_deg=0)
print(f'\nbaseline (author defaults)      CV = {rmse(predict(**BASE)):.4f}')

best = dict(BASE); best_cv = rmse(predict(**BASE))
# coordinate descent: cheap, and the surface is unlikely to interact strongly
GRID = [
    ('lp_key', [k for k in LP] or ['likpf_scale_5']),
    ('w_surf', [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6]),
    ('w_sub1', [0.3, 0.45, 0.6, 0.75, 0.9, 1.0]),
    ('alpha', [0.4, 0.55, 0.7, 0.85, 1.0, 1.15]),
    ('tau', [1e-9, 30.0, 60.0, 85.0, 150.0, 300.0, 600.0]),
    ('w_pf', [0.0, 0.1, 0.2]),
    ('trend_deg', [0, 1, 2, 3, 4]),
    ('sg_win', [61, 241, 401, 801, 1601, 3201]),
]
for rnd in range(3):
    improved = False
    for key, vals in GRID:
        cur = best[key]
        for v in vals:
            if v == cur:
                continue
            trial = dict(best); trial[key] = v
            cv = rmse(predict(**trial))
            if cv < best_cv - 1e-4:
                best_cv, best, improved = cv, trial, True
        print(f'  round{rnd} {key:8s} -> {best[key]}   CV={best_cv:.4f}', flush=True)
    if not improved:
        break

print(f'\nTUNED CV = {best_cv:.4f}   (baseline {rmse(predict(**BASE)):.4f})')
print('params:', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()})
print(f'estimated LB = {best_cv - 0.81:.3f}   (calibration LB = CV - 0.81)')
import json
json.dump({k: (v if not isinstance(v, np.generic) else float(v)) for k, v in best.items()},
          open('pp_best.json', 'w'), indent=1)
print('wrote pp_best.json')
