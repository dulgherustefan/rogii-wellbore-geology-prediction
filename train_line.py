"""Learn the per-well TVT line (offset + slope) and score it on the well-level
CV harness, which reproduces the organiser's prefix/suffix mask exactly."""
import numpy as np, pandas as pd, json, sys
import lightgbm as lgb
from sklearn.model_selection import KFold

df = pd.read_pickle('wellfeat.pkl')
FEATS = [c for c in df.columns if c not in {'wid', 'y_off', 'y_slope'}]
print(f'wells={len(df)} feats={len(FEATS)}')

PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=15,
              min_child_samples=25, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=10.0, objective='regression',
              verbose=-1, n_jobs=-1)

X = df[FEATS].values
folds = list(KFold(5, shuffle=True, random_state=42).split(X))
oof = {}
for tgt in ('y_off', 'y_slope'):
    y = df[tgt].values
    o = np.zeros(len(y))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X[tr], y[tr])
        o[va] = m.predict(X[va])
    oof[tgt] = o
    ss = 1 - np.sum((y - o) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f'  {tgt:8s} OOF R2 = {ss:+.4f}   corr = {np.corrcoef(y, o)[0,1]:+.3f}')

pred = {w: (float(a), float(b)) for w, a, b in zip(df.wid, oof['y_off'], oof['y_slope'])}
json.dump(pred, open('line_oof.json', 'w'))
print('wrote line_oof.json')

# ── end-metric check on the CV harness ───────────────────────────────────
import cvharness as C
P = pred

# calibrate the OOF predictions: a weak, noisy signal must be shrunk to the
# amount it actually explains, otherwise it adds more variance than it removes
CAL = {}
for tgt in ('y_off', 'y_slope'):
    o, yv = oof[tgt], df[tgt].values
    a = float(np.cov(o, yv, bias=True)[0, 1] / max(o.var(), 1e-12))
    CAL[tgt] = a
    print(f'  optimal shrink for {tgt}: {a:.3f}')


def _line(hw, tw, wid, use_off=True, use_slope=True, cal=True):
    t = hw['TVT_input'].to_numpy(float); md = hw['MD'].to_numpy(float)
    k = np.flatnonzero(np.isfinite(t))
    t0, m0 = t[k[-1]], md[k[-1]]
    off, sl = P.get(wid, (0.0, 0.0))
    if cal:
        off *= CAL['y_off']; sl *= CAL['y_slope']
    out = t.copy(); ev = ~np.isfinite(t)
    out[ev] = t0 + (off if use_off else 0.0) + (sl if use_slope else 0.0) * (md[ev] - m0)
    return out


def make(**kw):
    f = lambda hw, tw, wid, _kw=kw: _line(hw, tw, wid, **_kw)
    f.needs_wid = True
    return f


for nm, kw in [('learned_off', dict(use_slope=False)),
               ('learned_slope', dict(use_off=False)),
               ('learned_line', dict()),
               ('learned_line_raw', dict(cal=False))]:
    C.CANDS[nm] = make(**kw)

C.run(names=['const', 'learned_off', 'learned_slope', 'learned_line',
             'learned_line_raw', 'ORC_tconst', 'ORC_tline'],
      n_wells=int(sys.argv[1]) if len(sys.argv) > 1 else 773, n_jobs=8)
