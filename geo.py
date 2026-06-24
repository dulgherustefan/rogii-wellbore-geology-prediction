"""Geosteering TVT inversion — banded vectorized Viterbi.
Emission: GR level (+optional gradient) mismatch vs typewell + anchor pull.
Transition: quadratic smoothness within a band (allows moderate dip / small faults).
"""
import numpy as np


def smooth(x, k):
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode='same')


def align(md, gr_obs, tvt_input, tw_tvt, tw_gr,
          grid_res=0.5, band=45.0, smooth_k=7,
          lam=0.5, s_lvl=10.0, max_step=3.0,
          w_grad=0.0, s_grad=6.0,
          gamma=0.003, gamma_relax=2500.0,
          calib=True):
    tvt_input = tvt_input.astype(float)
    pred = tvt_input.copy()
    eval_mask = ~np.isfinite(tvt_input)
    if not eval_mask.any():
        return pred
    cut = int(np.argmax(eval_mask))
    anchor = tvt_input[cut - 1] if cut > 0 else float(np.median(tw_tvt))

    order = np.argsort(tw_tvt)
    tt, tg = tw_tvt[order], tw_gr[order]
    grid = np.arange(anchor - band, anchor + band + grid_res, grid_res)
    G = len(grid)
    ref = np.interp(grid, tt, tg, left=tg[0], right=tg[-1])
    dref = np.gradient(ref, grid_res)

    ki = np.where(np.isfinite(tvt_input))[0]
    a, b = 1.0, 0.0
    if calib and len(ki) > 30:
        kref = np.interp(tvt_input[ki], grid, ref)
        A = np.vstack([kref, np.ones(len(kref))]).T
        sol, *_ = np.linalg.lstsq(A, gr_obs[ki], rcond=None)
        if 0.2 < sol[0] < 5:
            a, b = sol
    ref_cal = a * ref + b
    dref_cal = a * dref

    gr = gr_obs.copy()
    gr = np.where(np.isfinite(gr), gr, np.nanmedian(gr))
    grs = smooth(gr, smooth_k)
    dgr = np.gradient(grs)

    ev = np.where(eval_mask)[0]
    m = len(ev)
    grv = grs[ev]
    have = np.isfinite(gr_obs[ev])

    emis = (grv[:, None] - ref_cal[None, :]) ** 2 / (2 * s_lvl ** 2)
    if w_grad > 0:
        emis = emis + w_grad * (dgr[ev][:, None] - dref_cal[None, :]) ** 2 / (2 * s_grad ** 2)
    emis[~have, :] = 0.0
    relax = np.exp(-np.arange(m) / gamma_relax)
    emis = emis + gamma * relax[:, None] * (grid[None, :] - anchor) ** 2

    mj = max(1, int(round(max_step / grid_res)))
    offs = np.arange(-mj, mj + 1)
    tpen = lam * (offs * grid_res) ** 2
    INF = 1e17

    cost = emis[0] + lam * (grid - anchor) ** 2  # start near anchor
    back = np.empty((m, G), dtype=np.int32)
    back[0] = np.arange(G)
    ar = np.arange(G)
    for i in range(1, m):
        new = np.full(G, INF)
        bsrc = np.zeros(G, dtype=np.int32)
        for o, pen in zip(offs, tpen):
            src = np.roll(cost, o)
            if o > 0:
                src[:o] = INF
            elif o < 0:
                src[o:] = INF
            cand = src + pen
            u = cand < new
            new[u] = cand[u]
            bsrc[u] = ar[u] - o
        cost = new + emis[i]
        back[i] = bsrc
    path = np.empty(m, dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for i in range(m - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    pred[ev] = grid[path]
    return pred
