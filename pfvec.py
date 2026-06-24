"""Vectorised particle-filter ensemble: all seeds processed together as an
(S, N) array so the Python loop runs once over MD steps (not S times).
~S x fewer Python iterations than the per-seed loop. Returns likelihood-weighted
ensemble prediction matching pf.pf_ensemble semantics."""
import numpy as np, pandas as pd

def build_own_ref(ti, GR, tw_tvt, tw_gr, res=0.5, pad=40):
    kn = np.isfinite(ti) & np.isfinite(GR); tk = ti[kn]; gk = GR[kn]
    if len(tk) < 30:
        return tw_tvt, tw_gr
    lo = min(tk.min(), tw_tvt.min())-pad; hi = max(tk.max(), tw_tvt.max())+pad
    grid = np.arange(lo, hi, res); tw_on = np.interp(grid, tw_tvt, tw_gr)
    bins = np.clip(np.round((tk-lo)/res).astype(int), 0, len(grid)-1)
    sv = np.zeros(len(grid)); cnt = np.zeros(len(grid)); np.add.at(sv, bins, gk); np.add.at(cnt, bins, 1)
    ref = np.full(len(grid), np.nan); ref[cnt > 0] = sv[cnt > 0]/cnt[cnt > 0]; have = np.isfinite(ref)
    if have.sum() > 20:
        A = np.vstack([tw_on[have], np.ones(have.sum())]).T; sol, *_ = np.linalg.lstsq(A, ref[have], rcond=None)
        a, b = sol if 0.2 < sol[0] < 5 else (1.0, 0.0); ref[~have] = (a*tw_on+b)[~have]
    else:
        ref[~have] = tw_on[~have]
    ref = pd.Series(ref).interpolate().bfill().ffill().rolling(3, min_periods=1, center=True).mean().values
    return grid, ref

def pf_ensemble_vec(hw, tw, n_seeds=28, N=500, own_ref=False, scale=5.0, init_spread=4.5,
                    MOM=0.998, VN=0.002, PN=0.005, RP=0.1, RR=0.001, RESAMP=0.5, seed0=0):
    tw_s = tw.sort_values('TVT'); tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    if own_ref:
        tw_tvt, tw_gr = build_own_ref(hw['TVT_input'].values.astype(float), hw['GR'].values.astype(float), tw_tvt, tw_gr)
    ti = hw['TVT_input'].values.astype(float); kn = np.isfinite(ti); ev = ~kn; out = ti.copy()
    if ev.sum() == 0 or kn.sum() < 3:
        if ev.any(): out[ev] = ti[kn][-1] if kn.any() else float(np.median(tw_tvt))
        return out
    Z = hw['Z'].values.astype(float); MD = hw['MD'].values.astype(float)
    GR = hw['GR'].interpolate(limit_direction='both').fillna(np.mean(tw_gr)).values.astype(float)
    ki = np.where(kn)[0]; ei = np.where(ev)[0]; last = ki[-1]
    tw_at_k = np.interp(ti[ki], tw_tvt, tw_gr); gs = float(np.clip(np.nanstd(GR[ki]-tw_at_k), 10., 60.))
    tail = ki[-30:]; dS = np.diff(ti[tail]+Z[tail]); dm = np.diff(MD[tail]); mk = dm > 0
    ir = float(np.median(dS[mk]/dm[mk])) if mk.sum() >= 3 else 0.0
    S = n_seeds
    rng = np.random.default_rng(1234+seed0)
    pos = (ti[last]+Z[last]) + init_spread*rng.standard_normal((S, N))
    rate = ir + 0.01*rng.standard_normal((S, N))
    w = np.full((S, N), 1.0/N)
    ll = np.zeros(S)
    res = np.empty((S, len(ei))); prev = MD[last]
    lo, hi = tw_tvt[0]-100, tw_tvt[-1]+100
    for i, gi in enumerate(ei):
        dms = max(MD[gi]-prev, 1.0)
        rate = MOM*rate + VN*rng.standard_normal((S, N))
        pos = pos + rate*dms + PN*rng.standard_normal((S, N))
        tvt_p = np.clip(pos - Z[gi], lo, hi); pos = tvt_p + Z[gi]
        eg = np.interp(tvt_p.ravel(), tw_tvt, tw_gr).reshape(S, N)
        d = (GR[gi]-eg)/gs
        lk = np.maximum(np.exp(-0.5*np.minimum(d*d, 600.)), 1e-300)
        ll += np.log(np.maximum((w*lk).sum(1), 1e-300))
        w = w*lk; ws = w.sum(1, keepdims=True); w = np.where(ws > 0, w/ws, 1.0/N)
        neff = 1.0/(w*w).sum(1)
        need = neff < RESAMP*N
        if need.any():
            for s in np.where(need)[0]:
                cum = np.cumsum(w[s]); u0 = rng.uniform(0, 1.0/N)
                idx = np.clip(np.searchsorted(cum, u0+np.arange(N)/N), 0, N-1)
                pos[s] = pos[s, idx] + RP*rng.standard_normal(N)
                rate[s] = rate[s, idx] + RR*rng.standard_normal(N)
                w[s] = 1.0/N
        res[:, i] = (w*(pos - Z[gi])).sum(1)
        prev = MD[gi]
    wn = np.exp((ll-ll.max())/scale); wn /= wn.sum()
    out[ei] = (wn[:, None]*res).sum(0)
    return out
