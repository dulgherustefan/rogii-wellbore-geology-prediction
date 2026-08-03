"""Sweep the particle-filter dynamics -- the one component never validated locally.

MOM/VN/PN, the 4.5 ft initial spread and the 10-60 GR-sigma clip were all
inherited from the public lineage and shipped unexamined. The lik-PF leg carries
0.40 of the post-process weight directly and sits at RMSE ~12.45, so it is worth
checking whether those defaults are actually right for this data.

Relative comparison only: 48 seeds keeps each config cheap, and every config is
scored on the identical well sample.
"""
import numpy as np, pandas as pd, glob, os, sys, time, itertools
from joblib import Parallel, delayed

DATA = os.environ.get('ROGII_DATA', 'data')
DEF = dict(MOM=0.998, VN=0.002, PN=0.005, RP=0.1, RR=0.001, RESAMP=0.5,
           SPREAD=4.5, RATE_SD=0.01, GS_LO=10.0, GS_HI=60.0, GS_LIST=None)


def _pf_once(md_v, z_v, gr_v, tw_tvt, tw_gr, gs, last_tvt, last_Z, last_MD, ir,
             N, seed, P):
    rng = np.random.default_rng(seed)
    pos = last_tvt + last_Z + P['SPREAD'] * rng.standard_normal(N)
    rate = ir + P['RATE_SD'] * rng.standard_normal(N)
    w = np.ones(N) / N
    res = np.empty(len(md_v))
    prev_MD = last_MD
    log_lik = 0.0
    for i in range(len(md_v)):
        dm = max(md_v[i] - prev_MD, 1.0)
        rate = P['MOM'] * rate + P['VN'] * rng.standard_normal(N)
        pos = pos + rate * dm + P['PN'] * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]
        d = (gr_v[i] - np.interp(tvt_p, tw_tvt, tw_gr)) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d ** 2, 600.)), 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk
        s = w.sum()
        w = w / s if s > 0 else np.ones(N) / N
        if 1.0 / (w ** 2).sum() < P['RESAMP'] * N:
            cum = np.cumsum(w)
            idx = np.clip(np.searchsorted(cum, rng.uniform(0, 1.0 / N) + np.arange(N) / N),
                          0, N - 1)
            pos = pos[idx] + P['RP'] * rng.standard_normal(N)
            rate = rate[idx] + P['RR'] * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]
    return res, log_lik


def leg_for_well(wid, P, n_seeds=48, N=350, scale=5.0, diag=False):
    hw = pd.read_csv(f'{DATA}/train/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{DATA}/train/{wid}__typewell.csv')
    t = hw['TVT_input'].to_numpy(float); y = hw['TVT'].to_numpy(float)
    md = hw['MD'].to_numpy(float); z = hw['Z'].to_numpy(float)
    kn = np.flatnonzero(np.isfinite(t)); ev = np.flatnonzero(~np.isfinite(t))
    if len(kn) < 120 or len(ev) < 60:
        return None
    tws = tw.sort_values('TVT')
    tw_tvt = tws['TVT'].to_numpy(float)
    tw_gr = tws['GR'].fillna(tws['GR'].mean()).to_numpy(float)
    gr = hw['GR'].astype(float).interpolate(limit_direction='both') \
                 .fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    a = kn[-1]
    # match the kernel exactly: the GR noise scale is computed from RAW GR with
    # NaN->0, not from the interpolated curve. GR is ~43% missing, so those zeros
    # inflate the std and gs pins at the ceiling for most wells -- which is what
    # makes the shipped filter weakly-informed and therefore stable. Using a
    # "correct" gs sharpens the likelihood and it locks onto wrong modes
    # (measured: leg 14.16 vs 11.81).
    raw_gr = hw['GR'].to_numpy(float)
    gs = float(np.clip(np.nanstd(np.nan_to_num(raw_gr[kn], nan=0.0)
                                 - np.interp(t[kn], tw_tvt, tw_gr)),
                       P['GS_LO'], P['GS_HI']))
    tail = kn[-30:]
    dt = np.diff(t[tail]); dz = np.diff(z[tail]); dm = np.diff(md[tail])
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    # The gs response is spiky, not smooth: 45 beats its own neighbours 40 and 50
    # by ~0.7 pooled CV on holdout, because a small change in gs flips which mode
    # the filter locks onto per well. Rather than bet on one point of a chaotic
    # surface, spread the seeds across a list of gs values -- an ensemble over the
    # nuisance parameter at identical cost.
    gs_list = P.get('GS_LIST') or [gs]
    preds, liks = [], []
    for s in range(n_seeds):
        gs_s = float(gs_list[s % len(gs_list)])
        p, ll = _pf_once(md[ev], z[ev], gr[ev], tw_tvt, tw_gr, gs_s,
                         float(t[a]), float(z[a]), float(md[a]), ir, N, s, P)
        preds.append(p); liks.append(ll)
    liks = np.array(liks)
    A = np.stack(preds, 0)
    # Log-likelihoods are NOT comparable across different gs: larger gs shrinks
    # d^2 and inflates log-lik, so a single softmax would collapse onto the
    # largest gs. Weight within each gs group, then average the groups equally.
    ng = len(gs_list)
    if ng > 1:
        parts = []
        for gi in range(ng):
            sel = np.arange(gi, len(liks), ng)
            lk = liks[sel] - liks[sel].max()
            wg = np.exp(lk / scale); wg /= wg.sum()
            parts.append((wg[:, None] * A[sel]).sum(0))
        est = np.mean(parts, axis=0)
    else:
        lk = liks - liks.max()
        w = np.exp(lk / scale); w /= w.sum()
        est = (w[:, None] * A).sum(0)
    if diag:
        # divergence diagnostics available at inference: how much the seeds
        # disagree, how peaked the likelihood weights are, and how far the
        # estimate wanders from the exactly-known anchor
        spread = float(np.mean(A.std(0)))
        n_eff = float(1.0 / np.sum(w ** 2)) if ng == 1 else float('nan')
        drift = float(np.max(np.abs(est - t[a])))
        return est - y[ev], dict(spread=spread, n_eff=n_eff, drift=drift,
                                 n_eval=len(ev), wid=wid)
    return est - y[ev]


def score(wids, P, n_jobs=6):
    r = [x for x in Parallel(n_jobs=n_jobs)(delayed(leg_for_well)(w, P) for w in wids)
         if x is not None]
    return float(np.sqrt(np.nanmean(np.concatenate(r) ** 2)))


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    wids = sorted(p.split('/')[-1].split('__')[0]
                  for p in glob.glob(f'{DATA}/train/*__horizontal_well.csv'))
    rng = np.random.RandomState(1)
    sub = list(rng.permutation(wids)[:n])
    t0 = time.time()
    base = score(sub, DEF)
    print(f'baseline (shipped defaults) = {base:.4f}   ({time.time()-t0:.0f}s)', flush=True)
    grid = [('SPREAD', [1.5, 3.0, 8.0, 15.0]),
            ('MOM', [0.99, 0.995, 0.9995]),
            ('VN', [0.0005, 0.001, 0.004, 0.008]),
            ('PN', [0.002, 0.01, 0.02]),
            ('GS_HI', [30.0, 45.0, 90.0]),
            ('RATE_SD', [0.003, 0.03])]
    for k, vals in grid:
        for v in vals:
            P = dict(DEF); P[k] = v
            s = score(sub, P)
            flag = '  <-- better' if s < base else ''
            print(f'  {k}={v!r:8s} {s:.4f}  (delta {s-base:+.4f}){flag}', flush=True)
