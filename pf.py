"""Particle filter for TVT tracking. State = S = TVT+Z (structural level),
evolved as momentum random walk along MD, GR likelihood vs typewell.
tvt = S - Z. Likelihood-weighted multi-seed ensemble."""
import numpy as np, pandas as pd

def build_own_ref(ti, GR, tw_tvt, tw_gr, res=0.5, pad=40):
    """GR(TVT) reference from the well's own known zone (high-res), typewell-filled."""
    kn = np.isfinite(ti) & np.isfinite(GR)
    tk = ti[kn]; gk = GR[kn]
    if len(tk) < 30:
        return tw_tvt, tw_gr
    lo = min(tk.min(), tw_tvt.min())-pad; hi = max(tk.max(), tw_tvt.max())+pad
    grid = np.arange(lo, hi, res)
    tw_on = np.interp(grid, tw_tvt, tw_gr)
    bins = np.clip(np.round((tk-lo)/res).astype(int), 0, len(grid)-1)
    sv = np.zeros(len(grid)); cnt = np.zeros(len(grid))
    np.add.at(sv, bins, gk); np.add.at(cnt, bins, 1)
    ref = np.full(len(grid), np.nan); ref[cnt > 0] = sv[cnt > 0]/cnt[cnt > 0]
    have = np.isfinite(ref)
    if have.sum() > 20:
        A = np.vstack([tw_on[have], np.ones(have.sum())]).T
        sol, *_ = np.linalg.lstsq(A, ref[have], rcond=None)
        a, b = sol if 0.2 < sol[0] < 5 else (1.0, 0.0)
        ref[~have] = (a*tw_on+b)[~have]
    else:
        ref[~have] = tw_on[~have]
    import pandas as _pd
    ref = _pd.Series(ref).interpolate().bfill().ffill().rolling(3, min_periods=1, center=True).mean().values
    return grid, ref

def run_pf(hw, tw, n_particles=500, seed=42, init_spread=4.5,
           MOM=0.998, VN=0.002, PN=0.005, RP=0.1, RR=0.001, RESAMP=0.5,
           gr_sig_clip=(10.,60.), ref=None):
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    if ref is not None:
        tw_tvt, tw_gr = ref
    ti = hw['TVT_input'].values.astype(float)
    kn_mask = np.isfinite(ti); ev_mask = ~kn_mask
    out = ti.copy()
    if ev_mask.sum() == 0:
        return out, 0.0
    Z = hw['Z'].values.astype(float); MD = hw['MD'].values.astype(float)
    GR = hw['GR'].interpolate(limit_direction='both')
    GR = GR.fillna(tw_gr.mean()).values.astype(float)
    kn_idx = np.where(kn_mask)[0]; ev_idx = np.where(ev_mask)[0]
    last = kn_idx[-1]
    last_tvt = ti[last]; last_Z = Z[last]; last_MD = MD[last]
    # GR sigma from known-zone residual vs typewell
    tw_at_k = np.interp(ti[kn_idx], tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(GR[kn_idx]-tw_at_k), *gr_sig_clip))
    # initial structural rate dS/dMD from recent tail
    tail = kn_idx[-30:]
    dS = np.diff(ti[tail]+Z[tail]); dm = np.diff(MD[tail])
    m = dm > 0
    ir = float(np.median(dS[m]/dm[m])) if m.sum() >= 3 else 0.0

    N = n_particles; rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + init_spread*rng.standard_normal(N)
    rate = ir + 0.01*rng.standard_normal(N)
    w = np.ones(N)/N
    res = np.empty(len(ev_idx)); prev_MD = last_MD; log_lik = 0.0
    for i, gi in enumerate(ev_idx):
        dm_step = max(MD[gi]-prev_MD, 1.0)
        rate = MOM*rate + VN*rng.standard_normal(N)
        pos = pos + rate*dm_step + PN*rng.standard_normal(N)
        tvt_p = pos - Z[gi]
        tvt_p = np.clip(tvt_p, tw_tvt[0]-100, tw_tvt[-1]+100)
        pos = tvt_p + Z[gi]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (GR[gi]-eg)/gs
        lk = np.exp(-0.5*np.minimum(d**2, 600.)); lk = np.maximum(lk, 1e-300)
        log_lik += np.log(max(float((w*lk).sum()), 1e-300))
        w = w*lk; ws = w.sum(); w = w/ws if ws > 0 else np.ones(N)/N
        if 1.0/(w**2).sum() < RESAMP*N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0/N)
            idx = np.clip(np.searchsorted(cum, u0+np.arange(N)/N), 0, N-1)
            pos = pos[idx]+RP*rng.standard_normal(N)
            rate = rate[idx]+RR*rng.standard_normal(N)
            w = np.ones(N)/N
        res[i] = float(np.dot(w, pos-Z[gi]))
        prev_MD = MD[gi]
    out[ev_idx] = res
    return out, log_lik

def pf_ensemble(hw, tw, n_seeds=64, n_particles=500, scale=5.0, own_ref=False, **kw):
    ref = None
    if own_ref:
        tw_s = tw.sort_values('TVT')
        ref = build_own_ref(hw['TVT_input'].values.astype(float), hw['GR'].values.astype(float),
                            tw_s['TVT'].values.astype(float), tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float))
    preds = []; liks = []
    for s in range(n_seeds):
        p, ll = run_pf(hw, tw, n_particles=n_particles, seed=s, ref=ref, **kw)
        preds.append(p); liks.append(ll)
    liks = np.array(liks); wn = np.exp((liks-liks.max())/scale); wn /= wn.sum()
    return (wn[:, None]*np.stack(preds, 0)).sum(0)
