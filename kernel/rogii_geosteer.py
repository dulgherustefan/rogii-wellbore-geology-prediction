# ROGII Wellbore Geology Prediction — TVT inversion
# Self-contained Kaggle kernel. Internet OFF. Reads /kaggle/input, writes submission.csv.
# Method: geosteering GR-signature inversion.
#   - Reference GR(TVT) built from the well's OWN pre-PS (known) gamma ray (high res),
#     filled with the typewell where the known zone doesn't cover (slide 9 of task).
#   - Coarse-grid dynamic-programming alignment of post-PS GR to that reference,
#     with curvature (2nd-order) smoothness so TVT can drift smoothly (slides 6-7),
#     anchored to the last known TVT, shrunk toward the anchor for robustness.
import os, glob
import numpy as np, pandas as pd

INP = '/kaggle/input/rogii-wellbore-geology-prediction'
if not os.path.isdir(INP):
    INP = 'data'  # local fallback

SHRINK = 0.75
PARAMS = dict(res=1.0, band=55.0, seg=40, vmax=6.0, vres=0.5,
              s_lvl=8.0, curv=3.0, gamma=0.02, gamma_relax=40.0)

def build_ref(tvt_known, gr_known, twt, twg, res=0.5, pad=60):
    fin = np.isfinite(tvt_known) & np.isfinite(gr_known)
    tk, gk = tvt_known[fin], gr_known[fin]
    if len(tk) < 20:
        grid = np.arange(twt.min()-pad, twt.max()+pad, res)
        return grid, np.interp(grid, twt, twg)
    lo = min(tk.min(), twt.min()) - pad
    hi = max(tk.max(), twt.max()) + pad
    grid = np.arange(lo, hi, res)
    tw_on = np.interp(grid, twt, twg)
    bins = np.clip(np.round((tk - lo) / res).astype(int), 0, len(grid)-1)
    sumv = np.zeros(len(grid)); cnt = np.zeros(len(grid))
    np.add.at(sumv, bins, gk); np.add.at(cnt, bins, 1)
    ref = np.full(len(grid), np.nan)
    ref[cnt > 0] = sumv[cnt > 0] / cnt[cnt > 0]
    have = np.isfinite(ref)
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

def align(md, gr_obs, tvt_input, twt, twg,
          res=1.0, band=55.0, seg=40, vmax=6.0, vres=0.5,
          s_lvl=8.0, curv=3.0, gamma=0.02, gamma_relax=40.0):
    tvt_input = tvt_input.astype(float)
    pred = tvt_input.copy()
    em = ~np.isfinite(tvt_input)
    if not em.any():
        return pred
    cut = int(np.argmax(em))
    anchor = tvt_input[cut-1] if cut > 0 else float(np.median(twt))
    grid_ref, ref = build_ref(tvt_input, gr_obs, twt, twg, res=0.5)
    ev = np.where(em)[0]
    gr = np.where(np.isfinite(gr_obs), gr_obs, np.nan)
    segs = [ev[i:i+seg] for i in range(0, len(ev), seg)]
    seg_gr = np.array([np.nanmean(gr[s]) if np.isfinite(gr[s]).any() else np.nan for s in segs])
    mc = len(segs)
    tgrid = np.arange(anchor-band, anchor+band+res, res); Gt = len(tgrid)
    vgrid = np.arange(-vmax, vmax+vres, vres); Gv = len(vgrid)
    ref_t = np.interp(tgrid, grid_ref, ref)
    emis = np.zeros((mc, Gt))
    for i in range(mc):
        if np.isfinite(seg_gr[i]):
            emis[i] = (seg_gr[i]-ref_t)**2/(2*s_lvl**2)
    relax = np.exp(-np.arange(mc)/gamma_relax)
    emis += gamma*relax[:, None]*(tgrid-anchor)**2
    INF = 1e18
    cost = np.full((Gt, Gv), INF)
    for vi, v in enumerate(vgrid):
        cost[:, vi] = emis[0] + 0.5*(tgrid-anchor)**2 + 3.0*v**2
    back = np.empty((mc, Gt, Gv, 2), dtype=np.int32)
    tidx = np.arange(Gt)
    vstep = np.round(vgrid/res).astype(int)
    nt_all = tidx[:, None] + vstep[None, :]
    valid_all = (nt_all >= 0) & (nt_all < Gt)
    for i in range(1, mc):
        new = np.full((Gt, Gv), INF)
        bk = np.zeros((Gt, Gv, 2), dtype=np.int32)
        for vi in range(Gv):
            ntv = nt_all[:, vi]; ok = valid_all[:, vi]
            base = cost[:, vi]
            for vpi in range(Gv):
                pen = curv*(vgrid[vpi]-vgrid[vi])**2
                cand = base + pen
                tgt = ntv[ok]; cc = cand[ok]
                cur = new[tgt, vpi]
                upd = cc < cur
                gi = np.where(ok)[0][upd]
                new[ntv[gi], vpi] = cc[upd]
                bk[ntv[gi], vpi, 0] = gi
                bk[ntv[gi], vpi, 1] = vi
        new += emis[i][:, None]
        cost = new; back[i] = bk
    flat = np.argmin(cost); t, v = np.unravel_index(flat, cost.shape)
    seg_tvt = np.empty(mc)
    for i in range(mc-1, -1, -1):
        seg_tvt[i] = tgrid[t]
        pt, pv = back[i, t, v]; t, v = pt, pv
    seg_centers = np.array([np.mean(md[s]) for s in segs])
    full = np.interp(md[ev], seg_centers, seg_tvt)
    pred[ev] = anchor + SHRINK*(full - anchor)
    return pred

def main():
    sub = pd.read_csv(f'{INP}/sample_submission.csv')
    sub['well'] = sub['id'].str.split('_').str[0]
    sub['idx'] = sub['id'].str.split('_').str[1].astype(int)
    out = {}
    wells = sorted(sub['well'].unique())
    for n, wid in enumerate(wells):
        h = pd.read_csv(f'{INP}/test/{wid}__horizontal_well.csv')
        tw = pd.read_csv(f'{INP}/test/{wid}__typewell.csv').dropna(subset=['TVT', 'GR']).sort_values('TVT')
        ti = h['TVT_input'].values.astype(float)
        try:
            pred = align(h['MD'].values, h['GR'].values, ti, tw['TVT'].values, tw['GR'].values, **PARAMS)
        except Exception as e:
            anchor = ti[np.isfinite(ti)][-1] if np.isfinite(ti).any() else float(tw['TVT'].median())
            pred = np.where(np.isfinite(ti), ti, anchor)
        g = sub[sub['well'] == wid]
        for idx in g['idx'].values:
            out[f'{wid}_{idx}'] = pred[idx] if idx < len(pred) else (ti[np.isfinite(ti)][-1] if np.isfinite(ti).any() else 0.0)
        if (n+1) % 50 == 0:
            print(f'{n+1}/{len(wells)} wells done', flush=True)
    sub['tvt'] = sub['id'].map(out)
    # safety: fill any remaining NaN with per-well last known
    sub['tvt'] = sub['tvt'].fillna(sub['tvt'].median())
    sub[['id', 'tvt']].to_csv('submission.csv', index=False)
    print('wrote submission.csv', len(sub), 'rows; tvt NaNs:', sub['tvt'].isna().sum())

if __name__ == '__main__':
    main()
