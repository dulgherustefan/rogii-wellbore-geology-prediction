"""Aligner v3: reference = own pre-PS GR(TVT) + typewell fill. DP inversion."""
import numpy as np, pandas as pd

def build_ref(tvt_known, gr_known, twt, twg, res=0.5, pad=60):
    fin = np.isfinite(tvt_known) & np.isfinite(gr_known)
    tk, gk = tvt_known[fin], gr_known[fin]
    lo = min(tk.min(), twt.min()) - pad if len(tk) else twt.min()-pad
    hi = max(tk.max(), twt.max()) + pad if len(tk) else twt.max()+pad
    grid = np.arange(lo, hi, res)
    tw_on = np.interp(grid, twt, twg)
    if len(tk) < 20:
        return grid, tw_on
    bins = np.round((tk - lo) / res).astype(int)
    bins = np.clip(bins, 0, len(grid)-1)
    ref = np.full(len(grid), np.nan)
    sumv = np.zeros(len(grid)); cnt = np.zeros(len(grid))
    np.add.at(sumv, bins, gk); np.add.at(cnt, bins, 1)
    ref[cnt > 0] = sumv[cnt > 0] / cnt[cnt > 0]
    have = np.isfinite(ref)
    # calibrate typewell to own ref where both exist
    if have.sum() > 20:
        A = np.vstack([tw_on[have], np.ones(have.sum())]).T
        sol, *_ = np.linalg.lstsq(A, ref[have], rcond=None)
        a, b = sol if 0.2 < sol[0] < 5 else (1.0, 0.0)
        twcal = a*tw_on + b
    else:
        twcal = tw_on
    ref[~have] = twcal[~have]
    ref = pd.Series(ref).interpolate().bfill().ffill().rolling(3, min_periods=1, center=True).mean().values
    return grid, ref

def smooth(x, k):
    if k <= 1: return x
    return np.convolve(x, np.ones(k)/k, mode='same')

def align(md, gr_obs, tvt_input, twt, twg,
          res=0.5, band=50.0, smooth_k=7, lam=0.6, s_lvl=9.0,
          max_step=2.0, gamma=0.01, gamma_relax=3000.0):
    tvt_input = tvt_input.astype(float)
    pred = tvt_input.copy()
    em = ~np.isfinite(tvt_input)
    if not em.any(): return pred
    cut = int(np.argmax(em))
    anchor = tvt_input[cut-1] if cut>0 else float(np.median(twt))
    grid_ref, ref = build_ref(tvt_input, gr_obs, twt, twg, res=res)

    lo = anchor-band; hi = anchor+band
    grid = np.arange(lo, hi+res, res); G=len(grid)
    ref_g = np.interp(grid, grid_ref, ref)

    gr = np.where(np.isfinite(gr_obs), gr_obs, np.nanmedian(gr_obs))
    grs = smooth(gr, smooth_k)
    ev = np.where(em)[0]; m=len(ev)
    have = np.isfinite(gr_obs[ev])
    emis = (grs[ev][:,None]-ref_g[None,:])**2/(2*s_lvl**2)
    emis[~have,:] = 0.0
    relax = np.exp(-np.arange(m)/gamma_relax)
    emis += gamma*relax[:,None]*(grid[None,:]-anchor)**2

    mj=max(1,int(round(max_step/res))); offs=np.arange(-mj,mj+1); tpen=lam*(offs*res)**2
    INF=1e17
    cost=emis[0]+lam*(grid-anchor)**2
    back=np.empty((m,G),dtype=np.int32); back[0]=np.arange(G); ar=np.arange(G)
    for i in range(1,m):
        new=np.full(G,INF); bsrc=np.zeros(G,dtype=np.int32)
        for o,pen in zip(offs,tpen):
            src=np.roll(cost,o)
            if o>0: src[:o]=INF
            elif o<0: src[o:]=INF
            cand=src+pen; u=cand<new; new[u]=cand[u]; bsrc[u]=ar[u]-o
        cost=new+emis[i]; back[i]=bsrc
    path=np.empty(m,dtype=np.int32); path[-1]=int(np.argmin(cost))
    for i in range(m-1,0,-1): path[i-1]=back[i,path[i]]
    pred[ev]=grid[path]
    return pred
