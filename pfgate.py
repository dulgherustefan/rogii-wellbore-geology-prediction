"""Detect PF divergence and fall back, instead of tuning the PF.

The holdout run reframed the problem. The retuned 'combo' config improves the
TYPICAL well a lot (median well-RMSE 7.372 -> 5.802, trim90 7.676 -> 6.677) yet
its pooled RMSE is unchanged (11.695 -> 11.681), because pooled RMSE is dominated
by a handful of wells where the filter locks onto the wrong marker and runs away.
Median 5.8 against pooled 11.7 is the whole story.

So the lever is not better dynamics, it is knowing WHEN the tracker has diverged.
Three diagnostics are available at inference without any label:
  spread : mean disagreement between seeds
  n_eff  : how peaked the likelihood weights are across seeds
  drift  : furthest excursion from the exactly-known anchor
This measures whether any of them predicts the per-well error, and what a gated
fallback to the anchor is worth.
"""
import numpy as np, pandas as pd, glob, sys, time, json
from joblib import Parallel, delayed
import pfsweep as S

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
COMBO = dict(MOM=0.9995, RATE_SD=0.003, SPREAD=1.5, PN=0.01, VN=0.001)


def one(wid):
    hw = pd.read_csv(f'data/train/{wid}__horizontal_well.csv')
    t = hw['TVT_input'].to_numpy(float)
    kn = np.flatnonzero(np.isfinite(t))
    if len(kn) == 0:
        return None
    anchor = float(t[kn[-1]])
    out = {}
    for tag, over in (('base', {}), ('combo', COMBO)):
        P = dict(S.DEF); P.update(over)
        r = S.leg_for_well(wid, P, SEEDS, 350, 5.0, diag=True)
        if r is None:
            return None
        err, dg = r
        out[tag] = (err, dg)
    y = hw['TVT'].to_numpy(float)[np.flatnonzero(~np.isfinite(t))]
    return wid, out, anchor, y


if __name__ == '__main__':
    t0 = time.time()
    wids = sorted(p.split('/')[-1].split('__')[0]
                  for p in glob.glob('data/train/*__horizontal_well.csv'))
    rng = np.random.RandomState(1)
    sub = list(rng.permutation(wids)[100:100 + N])          # holdout half
    res = [r for r in Parallel(n_jobs=6)(delayed(one)(w) for w in sub) if r]
    print(f'wells={len(res)} seeds={SEEDS} ({time.time()-t0:.0f}s)', flush=True)

    rows = []
    for wid, out, anchor, y in res:
        for tag in ('base', 'combo'):
            err, dg = out[tag]
            rows.append(dict(wid=wid, cfg=tag, rmse=float(np.sqrt(np.nanmean(err ** 2))),
                             n=len(err), anchor_rmse=float(np.sqrt(np.nanmean((anchor - y) ** 2))),
                             **{k: v for k, v in dg.items() if k != 'wid'}))
    df = pd.DataFrame(rows)
    df.to_csv('pfgate.csv', index=False)

    for cfg in ('base', 'combo'):
        g = df[df.cfg == cfg]
        print(f'\n--- {cfg} ---')
        print(f'  pooled={np.sqrt(np.average(g.rmse**2, weights=g.n)):.3f}  '
              f'median={g.rmse.median():.3f}')
        for c in ('spread', 'n_eff', 'drift'):
            print(f'  corr(log {c}, log well-RMSE) = '
                  f'{np.corrcoef(np.log(g[c] + 1e-6), np.log(g.rmse + 1e-6))[0, 1]:+.3f}')

    # what a gated fallback to the anchor is worth, sweeping the drift threshold
    print('\ngated fallback to anchor when drift exceeds a threshold:')
    for cfg in ('base', 'combo'):
        g = df[df.cfg == cfg].reset_index(drop=True)
        base_p = np.sqrt(np.average(g.rmse ** 2, weights=g.n))
        best = (None, base_p)
        for thr in (20, 30, 40, 60, 80, 120, 200):
            use = np.where(g.drift > thr, g.anchor_rmse, g.rmse)
            p = np.sqrt(np.average(use ** 2, weights=g.n))
            n_gated = int((g.drift > thr).sum())
            if p < best[1]:
                best = (thr, p)
            print(f'  {cfg} thr={thr:4d}: pooled={p:.3f}  (gated {n_gated}/{len(g)} wells)')
        print(f'  -> {cfg} best thr={best[0]} pooled {base_p:.3f} -> {best[1]:.3f}')
