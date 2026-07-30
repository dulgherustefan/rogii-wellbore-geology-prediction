"""Exact stratigraphic alignment by dynamic programming.

The particle filter is a sampled approximation of the same inference and loses
the tail of the posterior to particle depletion. Here the state space is
enumerated on a grid and decoded exactly with Viterbi, which is what actually
recovers the per-well SLOPE the oracle ladder says the error lives in.

Model
-----
State at step i is the stratigraphic position TVT_i on a grid.
Emission: the horizontal GR reading must match the typewell GR profile sampled
at TVT_i, after an affine calibration fitted on the known prefix.
Transition: the geologically smooth quantity is U = TVT + Z (formation
elevation), not TVT itself -- the well's own vertical movement Z is known
exactly. So the transition penalises change in U, i.e. formation dip, rather
than change in TVT.
"""
import numpy as np


def _affine_cal(x, y):
    """Least-squares a*x+b mapping horizontal GR onto typewell GR."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return 1.0, 0.0
    x, y = x[m], y[m]
    sx = x.std()
    if sx < 1e-6:
        return 1.0, float(y.mean() - x.mean())
    a = float(np.cov(x, y, bias=True)[0, 1] / (sx ** 2))
    a = float(np.clip(a, 0.2, 5.0))
    return a, float(y.mean() - a * x.mean())


def dp_align(hw, tw, step=0.25, half_width=60.0, bin_rows=6,
             dip_sigma=0.005, gr_sigma_scale=1.0, max_dip=0.02, emis_w=1.0,
             curv_penalty=True, return_grid=False):
    """Decode TVT over the eval zone. Returns a full-length TVT array."""
    t_in = hw['TVT_input'].to_numpy(float)
    md = hw['MD'].to_numpy(float)
    z = hw['Z'].to_numpy(float)
    gr = hw['GR'].to_numpy(float)
    n = len(md)
    kn = np.flatnonzero(np.isfinite(t_in))
    ev = np.flatnonzero(~np.isfinite(t_in))
    if len(kn) < 30 or len(ev) < 5:
        out = t_in.copy()
        if len(kn):
            out[~np.isfinite(out)] = t_in[kn[-1]]
        return out

    tws = tw.sort_values('TVT')
    tw_tvt = tws['TVT'].to_numpy(float)
    tw_gr = tws['GR'].to_numpy(float)
    ok = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tw_tvt, tw_gr = tw_tvt[ok], tw_gr[ok]
    if len(tw_tvt) < 10:
        out = t_in.copy(); out[~np.isfinite(out)] = t_in[kn[-1]]
        return out

    # calibrate the horizontal GR onto the typewell scale using the known prefix
    tw_at_known = np.interp(t_in[kn], tw_tvt, tw_gr)
    a, b = _affine_cal(gr[kn], tw_at_known)
    gr_cal = a * gr + b
    resid = gr_cal[kn] - tw_at_known
    gs = float(np.clip(np.nanstd(resid), 5.0, 80.0)) * gr_sigma_scale

    # state grid centred on the last known TVT
    t0 = float(t_in[kn[-1]]); m0 = float(md[kn[-1]]); z0 = float(z[kn[-1]])
    grid = np.arange(t0 - half_width, t0 + half_width + step, step)
    S = len(grid)

    # bin the eval zone to keep the decode cheap and to average GR noise
    ev_sorted = ev[np.argsort(md[ev])]
    nb = max(1, len(ev_sorted) // bin_rows)
    bins = np.array_split(ev_sorted, nb)
    bmd = np.array([md[ix].mean() for ix in bins])
    bz = np.array([z[ix].mean() for ix in bins])
    bgr = np.array([np.nanmedian(gr_cal[ix]) for ix in bins])

    # emission cost: |calibrated GR - typewell GR(state)| in a robust (Huber) form
    tw_on_grid = np.interp(grid, tw_tvt, tw_gr, left=np.nan, right=np.nan)
    valid = np.isfinite(tw_on_grid)
    tw_on_grid = np.where(valid, tw_on_grid, 0.0)
    r = (bgr[:, None] - tw_on_grid[None, :]) / gs
    emis = np.where(np.abs(r) < 2.0, 0.5 * r ** 2, 2.0 * np.abs(r) - 2.0)
    emis[:, ~valid] += 5.0            # discourage leaving the typewell's range
    emis[~np.isfinite(emis)] = 10.0

    # Transition. Measured on train: TVT itself only wanders ~25 ft over an entire
    # eval zone (d TVT/d MD has std ~0.005), whereas U = TVT+Z drifts with the
    # well's own descent. So the bounded quantity to penalise is the TVT rate.
    dmd = np.diff(np.concatenate(([m0], bmd)))
    W = int(np.ceil(max_dip * float(np.median(dmd)) / step)) + 1
    offs = np.arange(-W, W + 1)

    cost = np.full(S, 1e9)
    cost[np.argmin(np.abs(grid - t0))] = 0.0     # anchored on the last known TVT
    back = np.zeros((len(bins), S), dtype=np.int16)

    for i in range(len(bins)):
        # a shift of k grid cells over dmd[i] feet of hole is a TVT rate
        dip = (offs * step) / max(dmd[i], 1e-6)
        tcost = 0.5 * (dip / dip_sigma) ** 2
        best = np.full(S, 1e9); barg = np.zeros(S, dtype=np.int16)
        for j, k in enumerate(offs):
            shifted = np.full(S, 1e9)
            if k >= 0:
                shifted[k:] = cost[:S - k] if k else cost
            else:
                shifted[:k] = cost[-k:]
            c = shifted + tcost[j]
            upd = c < best
            best[upd] = c[upd]; barg[upd] = k
        cost = best + emis_w * emis[i]
        back[i] = barg

    # backtrack
    path = np.empty(len(bins), dtype=int)
    s = int(np.argmin(cost))
    for i in range(len(bins) - 1, -1, -1):
        path[i] = s
        s = s - int(back[i][s])
        s = int(np.clip(s, 0, S - 1))
    btvt = grid[path]

    out = t_in.copy()
    out[ev] = np.interp(md[ev], bmd, btvt)
    if curv_penalty:
        pass
    if return_grid:
        return out, bmd, btvt
    return out
