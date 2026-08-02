"""Does a visible-prefix backtest predict a well's eval-zone level bias?

Error decomposition on the real OOF: pooled 9.19 = per-well level 7.08 (+) shape
5.86. Removing the level term perfectly would score 5.86; estimating it at
correlation rho gives sqrt(mean((e - rho*level)^2)) -- rho=0.5 -> 6.85,
rho=0.7 -> 6.24. The public leaderboard plateau at ~6.45 sits at rho ~= 0.6, so
per-well level estimation is the whole game and everything else is rounding.

The only honest level estimate available at inference is a backtest inside the
visible prefix: hide its tail, track into it, and measure the bias there. This
script measures whether that bias actually transfers to the eval zone.
"""
import numpy as np, pandas as pd, glob, os, time, sys
from joblib import Parallel, delayed

DATA = os.environ.get('ROGII_DATA', 'data')
CUTS = (0.50, 0.65, 0.75)


def _pf_once(md_v, z_v, gr_v, tw_tvt, tw_gr, gs, last_tvt, last_Z, last_MD, ir,
             N, seed):
    """One particle-filter sweep; identical dynamics to the kernel's tracker."""
    rng = np.random.default_rng(seed)
    pos = last_tvt + last_Z + 4.5 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    MOM, VN, PN, RP, RR, RESAMP = 0.998, 0.002, 0.005, 0.1, 0.001, 0.5
    res = np.empty(len(md_v))
    prev_MD = last_MD
    log_lik = 0.0
    for i in range(len(md_v)):
        dm = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]
        d = (gr_v[i] - np.interp(tvt_p, tw_tvt, tw_gr)) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d ** 2, 600.)), 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk
        s = w.sum()
        w = w / s if s > 0 else np.ones(N) / N
        if 1.0 / (w ** 2).sum() < RESAMP * N:
            cum = np.cumsum(w)
            idx = np.clip(np.searchsorted(cum, rng.uniform(0, 1.0 / N) + np.arange(N) / N),
                          0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]
    return res, log_lik


def pf_segment(hw, tw, known_idx, target_idx, n_seeds=24, N=350, scale=8.0):
    """Likelihood-weighted PF ensemble tracking from the end of known_idx across
    target_idx. Mirrors run_pf_lik_ensemble_scales at a single scale."""
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].to_numpy(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).to_numpy(float)
    t = hw['TVT_input'].to_numpy(float)
    md = hw['MD'].to_numpy(float); z = hw['Z'].to_numpy(float)
    gr = hw['GR'].astype(float).interpolate(limit_direction='both') \
                 .fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    a = known_idx[-1]
    tw_at_k = np.interp(t[known_idx], tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(np.nan_to_num(gr[known_idx]) - tw_at_k), 10., 60.))
    tail = known_idx[-30:]
    dt = np.diff(t[tail]); dz = np.diff(z[tail]); dm = np.diff(md[tail])
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    preds, liks = [], []
    for s in range(n_seeds):
        p, ll = _pf_once(md[target_idx], z[target_idx], gr[target_idx], tw_tvt, tw_gr,
                         gs, float(t[a]), float(z[a]), float(md[a]), ir, N, s)
        preds.append(p); liks.append(ll)
    liks = np.array(liks); liks -= liks.max()
    wts = np.exp(liks / scale); wts /= wts.sum()
    return (wts[:, None] * np.stack(preds, 0)).sum(0)


def well_row(wid, n_seeds=24, N=350):
    hw = pd.read_csv(f'{DATA}/train/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{DATA}/train/{wid}__typewell.csv')
    t = hw['TVT_input'].to_numpy(float)
    y = hw['TVT'].to_numpy(float)
    kn = np.flatnonzero(np.isfinite(t))
    ev = np.flatnonzero(~np.isfinite(t))
    if len(kn) < 120 or len(ev) < 60:
        return None
    out = {'well': wid, 'n_known': len(kn), 'n_eval': len(ev)}
    for f in CUTS:
        c = int(len(kn) * f)
        if c < 60 or len(kn) - c < 30:
            out[f'bias_{f}'] = np.nan; out[f'rmse_{f}'] = np.nan
            continue
        ki, ti = kn[:c], kn[c:]
        p = pf_segment(hw, tw, ki, ti, n_seeds=n_seeds, N=N)
        out[f'bias_{f}'] = float(np.mean(p - t[ti]))
        out[f'rmse_{f}'] = float(np.sqrt(np.mean((p - t[ti]) ** 2)))
        out[f'span_{f}'] = float(hw['MD'].to_numpy(float)[ti][-1] -
                                 hw['MD'].to_numpy(float)[ki][-1])
    # the quantity we are trying to predict
    p_ev = pf_segment(hw, tw, kn, ev, n_seeds=n_seeds, N=N)
    out['bias_eval'] = float(np.nanmean(p_ev - y[ev]))
    out['rmse_eval'] = float(np.sqrt(np.nanmean((p_ev - y[ev]) ** 2)))
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    t0 = time.time()
    wids = sorted(p.split('/')[-1].split('__')[0]
                  for p in glob.glob(f'{DATA}/train/*__horizontal_well.csv'))
    rng = np.random.RandomState(0)
    sub = list(rng.permutation(wids)[:n])
    # processes, not threads: the PF inner loop is 350-element numpy ops that
    # never release the GIL long enough to parallelise
    res = Parallel(n_jobs=6)(delayed(well_row)(w) for w in sub)
    df = pd.DataFrame([r for r in res if r is not None])
    df.to_csv('vpcal.csv', index=False)
    print(f'wells={len(df)} ({time.time()-t0:.0f}s)', flush=True)
    for f in CUTS:
        m = df[f'bias_{f}'].notna() & df['bias_eval'].notna()
        r = np.corrcoef(df.loc[m, f'bias_{f}'], df.loc[m, 'bias_eval'])[0, 1]
        print(f'  cut {f}: rho(bias_cal, bias_eval) = {r:+.3f}  n={m.sum()}')
    df['bias_avg'] = df[[f'bias_{f}' for f in CUTS]].mean(axis=1)
    m = df['bias_avg'].notna() & df['bias_eval'].notna()
    print(f'  AVG of cuts: rho = '
          f'{np.corrcoef(df.loc[m,"bias_avg"], df.loc[m,"bias_eval"])[0,1]:+.3f}')
