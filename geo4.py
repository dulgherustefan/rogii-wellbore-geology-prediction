"""Aligner v4: coarse-grid DP with curvature (2nd-order) smoothness via
augmented (TVT, velocity) state. Allows smooth slow drift, denoises GR by
segment-averaging. Reference = own pre-PS GR(TVT) + typewell fill."""
import numpy as np, pandas as pd
from geo3 import build_ref

def align(md, gr_obs, tvt_input, twt, twg,
          res=1.0, band=55.0, seg=40, vmax=6.0, vres=0.5,
          s_lvl=8.0, curv=2.0, gamma=0.02, gamma_relax=40.0):
    tvt_input = tvt_input.astype(float)
    pred = tvt_input.copy()
    em = ~np.isfinite(tvt_input)
    if not em.any(): return pred
    cut = int(np.argmax(em))
    anchor = tvt_input[cut-1] if cut>0 else float(np.median(twt))
    grid_ref, ref = build_ref(tvt_input, gr_obs, twt, twg, res=0.5)

    ev = np.where(em)[0]
    gr = np.where(np.isfinite(gr_obs), gr_obs, np.nan)
    # coarse segments over eval rows
    segs = [ev[i:i+seg] for i in range(0, len(ev), seg)]
    seg_gr = np.array([np.nanmean(gr[s]) if np.isfinite(gr[s]).any() else np.nan for s in segs])
    mc = len(segs)

    # TVT grid + velocity grid
    tgrid = np.arange(anchor-band, anchor+band+res, res); Gt=len(tgrid)
    vgrid = np.arange(-vmax, vmax+vres, vres); Gv=len(vgrid)
    ref_t = np.interp(tgrid, grid_ref, ref)

    # emission per segment x tvt
    emis = np.zeros((mc, Gt))
    for i in range(mc):
        if np.isfinite(seg_gr[i]):
            emis[i] = (seg_gr[i]-ref_t)**2/(2*s_lvl**2)
    # anchor pull (decaying)
    relax = np.exp(-np.arange(mc)/gamma_relax)
    emis += gamma*relax[:,None]*(tgrid-anchor)**2

    INF=1e18
    # cost[t_idx, v_idx]
    # init: first segment, start near anchor, velocity ~0
    ai = np.argmin(np.abs(tgrid-anchor))
    cost = np.full((Gt,Gv), INF)
    for vi,v in enumerate(vgrid):
        cost[:,vi] = emis[0] + 0.5*(tgrid-anchor)**2 + 3.0*v**2
    back = np.empty((mc,Gt,Gv,2), dtype=np.int32)
    vidx = np.arange(Gv); tidx=np.arange(Gt)
    vstep = vgrid/res  # velocity in grid units
    for i in range(1,mc):
        new = np.full((Gt,Gv), INF)
        bk = np.zeros((Gt,Gv,2),dtype=np.int32)
        # from each (t,v): next t' = t + round(v/res), v' free with curv penalty (v'-v)^2
        nt = tidx[:,None] + np.round(vstep)[None,:].astype(int)  # Gt x Gv next t index
        valid = (nt>=0)&(nt<Gt)
        for vi in range(Gv):
            for vpi in range(Gv):
                pen = curv*(vgrid[vpi]-vgrid[vi])**2
                # source states with velocity vi transitioning to velocity vpi
                src_cost = cost[:,vi] + pen  # Gt
                ntv = nt[:,vi]  # Gt
                ok = valid[:,vi]
                # scatter min into new[ntv, vpi]
                cand = src_cost[ok]
                tgt = ntv[ok]
                cur = new[tgt,vpi]
                upd = cand<cur
                idx = np.where(ok)[0][upd]
                new[ntv[idx],vpi]=cand[upd]
                bk[ntv[idx],vpi,0]=idx
                bk[ntv[idx],vpi,1]=vi
        new += emis[i][:,None]
        cost=new; back[i]=bk
    # backtrack
    flat=np.argmin(cost); ti0,vi0=np.unravel_index(flat,cost.shape)
    seg_tvt=np.empty(mc)
    t,v=ti0,vi0
    for i in range(mc-1,-1,-1):
        seg_tvt[i]=tgrid[t]
        pt,pv=back[i,t,v]
        t,v=pt,pv
    # interpolate seg_tvt (at segment centers) to full eval rows
    seg_centers=np.array([np.mean(md[s]) for s in segs])
    pred[ev]=np.interp(md[ev], seg_centers, seg_tvt)
    return pred
