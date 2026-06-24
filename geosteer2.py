"""Geosteering TVT inversion v2: fast Viterbi with quadratic transition
(Felzenszwalb generalized distance transform) + level & gradient GR emission
+ weak anchor pull. O(m*G) per well."""
import numpy as np


def gdt_1d(f, step, lam):
    """Generalized distance transform for quadratic: out[j]=min_i f[i]+lam*((j-i)*step)^2.
    Returns out and argmin index array. O(n)."""
    n = len(f)
    out = np.empty(n)
    arg = np.empty(n, dtype=np.int64)
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    s2 = (step * step) * lam
    for q in range(1, n):
        while True:
            p = v[k]
            s = ((f[q] + s2 * q * q) - (f[p] + s2 * p * p)) / (2 * s2 * (q - p))
            if s <= z[k]:
                k -= 1
            else:
                break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        p = v[k]
        out[q] = s2 * (q - p) * (q - p) + f[p]
        arg[q] = p
    return out, arg


def smooth(x, k):
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode='same')


def align(md, gr_obs, tvt_input, tw_tvt, tw_gr,
          grid_res=0.5, band=45.0, smooth_k=7,
          lam=0.6, s_lvl=10.0, w_grad=0.0, s_grad=6.0,
          gamma=0.004, gamma_relax=2500.0):
    """Return full TVT prediction. Known zone copied; eval zone solved."""
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

    # calibrate GR(obs) ~ a*ref+b on known zone
    ki = np.where(np.isfinite(tvt_input))[0]
    if len(ki) > 30:
        kref = np.interp(tvt_input[ki], grid, ref)
        A = np.vstack([kref, np.ones(len(kref))]).T
        sol, *_ = np.linalg.lstsq(A, gr_obs[ki], rcond=None)
        a, b = sol
        if not (0.2 < a < 5):
            a, b = 1.0, 0.0
    else:
        a, b = 1.0, 0.0
    ref_cal = a * ref + b
    dref_cal = a * dref

    gr = gr_obs.copy()
    med = np.nanmedian(gr)
    gr = np.where(np.isfinite(gr), gr, med)
    grs = smooth(gr, smooth_k)
    dgr = np.gradient(grs, 1.0)  # per-row gradient (proxy)

    ev = np.where(eval_mask)[0]
    m = len(ev)
    grv = grs[ev]
    have = np.isfinite(gr_obs[ev])

    # emission base (m x G)
    emis = (grv[:, None] - ref_cal[None, :]) ** 2 / (2 * s_lvl ** 2)
    if w_grad > 0:
        emis = emis + w_grad * (dgr[ev][:, None] - dref_cal[None, :]) ** 2 / (2 * s_grad ** 2)
    emis[~have, :] = 0.0
    # anchor pull, relaxing with distance into eval
    dist = (np.arange(m)) * 1.0
    relax = np.exp(-dist / gamma_relax)  # strong near anchor, fades
    apull = gamma * relax[:, None] * (grid[None, :] - anchor) ** 2
    emis = emis + apull

    INF = 1e17
    cost = emis[0].copy()
    back = np.empty((m, G), dtype=np.int64)
    back[0] = np.arange(G)
    for i in range(1, m):
        prop, arg = gdt_1d(cost, grid_res, lam)
        cost = prop + emis[i]
        back[i] = arg
    path = np.empty(m, dtype=np.int64)
    path[-1] = int(np.argmin(cost))
    for i in range(m - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    pred[ev] = grid[path]
    return pred
