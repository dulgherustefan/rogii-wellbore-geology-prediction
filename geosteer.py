"""Geosteering TVT inversion via constrained dynamic programming.

Target: predict TVT(MD) along the evaluation zone of a horizontal well,
matching observed GR to the typewell reference GR(TVT), anchored at the
known boundary and regularized for smoothness.
"""
import numpy as np
import pandas as pd


def smooth(x, k):
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode='same')


def build_typewell_lookup(tw, grid):
    """Interpolate typewell GR onto a regular TVT grid."""
    t = tw['TVT'].values
    g = tw['GR'].values
    order = np.argsort(t)
    t, g = t[order], g[order]
    return np.interp(grid, t, g, left=g[0], right=g[-1])


def calibrate_gr(known_gr, known_ref):
    """Fit GR_obs ~ a*GR_ref + b on known zone (robust-ish least squares)."""
    m = np.isfinite(known_gr) & np.isfinite(known_ref)
    if m.sum() < 20:
        return 1.0, 0.0
    A = np.vstack([known_ref[m], np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(A, known_gr[m], rcond=None)
    a, b = coef
    if not (0.2 < a < 5):  # guard against degenerate fit
        a, b = 1.0, 0.0
    return a, b


def dp_align(md, gr_obs, tvt_input, tw,
             grid_res=0.5, band=35.0, smooth_k=5,
             lam=8.0, drift_w=0.0, emis_sigma=12.0):
    """
    Returns full-length TVT prediction array.
    md, gr_obs, tvt_input: arrays for the whole well.
    Known zone where tvt_input is finite is copied through; eval zone solved by DP.
    """
    n = len(md)
    pred = tvt_input.copy().astype(float)
    eval_mask = ~np.isfinite(tvt_input)
    if not eval_mask.any():
        return pred
    cut = np.argmax(eval_mask)  # first eval index
    anchor_tvt = tvt_input[cut - 1] if cut > 0 else np.nanmedian(tw['TVT'].values)

    # TVT search grid centered on anchor, +/- band
    lo = anchor_tvt - band
    hi = anchor_tvt + band
    grid = np.arange(lo, hi + grid_res, grid_res)
    G = len(grid)
    ref_on_grid = build_typewell_lookup(tw, grid)  # GR_ref at each grid TVT

    # calibrate GR scale using known zone true tvt
    known_idx = np.where(np.isfinite(tvt_input))[0]
    if len(known_idx) > 30:
        kref = np.interp(tvt_input[known_idx], grid, ref_on_grid)
        a, b = calibrate_gr(gr_obs[known_idx], kref)
    else:
        a, b = 1.0, 0.0
    # transform reference into observed-GR space
    ref_cal = a * ref_on_grid + b

    gr = smooth(np.where(np.isfinite(gr_obs), gr_obs, np.nanmedian(gr_obs)), smooth_k)

    ev = np.where(eval_mask)[0]
    m = len(ev)
    # emission cost matrix (m x G): (gr - ref_cal)^2 / sigma^2
    grv = gr[ev][:, None]
    emis = (grv - ref_cal[None, :]) ** 2 / (2 * emis_sigma ** 2)
    # where GR missing, zero emission (rely on prior)
    miss = ~np.isfinite(gr_obs[ev])
    emis[miss, :] = 0.0

    # transition: penalize squared change in TVT (smoothness). lam scales per step.
    # banded: TVT can move at most `max_jump` grid cells per step
    max_jump = max(1, int(np.ceil(3.0 / grid_res)))  # up to 3 ft/step
    # precompute transition penalty for offsets
    offs = np.arange(-max_jump, max_jump + 1)
    trans_pen = lam * (offs * grid_res) ** 2

    INF = 1e18
    cost = np.full(G, INF)
    # start: anchored near anchor_tvt
    a_idx = int(round((anchor_tvt - lo) / grid_res))
    a_idx = min(max(a_idx, 0), G - 1)
    # initialize first eval step: emission + small penalty for distance from anchor
    start_pen = lam * (grid - anchor_tvt) ** 2
    cost = emis[0] + start_pen
    back = np.zeros((m, G), dtype=np.int32)

    for i in range(1, m):
        # for each target state j, best over source states j-offs
        new_cost = np.full(G, INF)
        best_src = np.zeros(G, dtype=np.int32)
        for o, pen in zip(offs, trans_pen):
            # source index = j - o
            src = np.roll(cost, o)  # src[j] = cost[j-o]
            # invalidate wrapped entries
            if o > 0:
                src[:o] = INF
            elif o < 0:
                src[o:] = INF
            cand = src + pen
            upd = cand < new_cost
            new_cost[upd] = cand[upd]
            best_src[upd] = (np.arange(G)[upd] - o)
        new_cost += emis[i]
        cost = new_cost
        back[i] = best_src

    # backtrack
    path = np.zeros(m, dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for i in range(m - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    pred[ev] = grid[path]
    return pred


def predict_well(h, tw, **kw):
    return dp_align(h['MD'].values, h['GR'].values, h['TVT_input'].values.astype(float), tw, **kw)
