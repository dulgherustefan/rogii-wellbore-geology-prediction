"""Structural dip from neighbouring wells -> the per-well TVT slope.

TVT = b_well + E(X,Y) - Z, so S = TVT + Z = b_well + E(X,Y). Within one well only
the along-path derivative of E is observable, but across several neighbouring
wells the full 2D gradient is identifiable once each well is allowed its own
datum b_well. Fitting S = b_i + a*X + c*Y jointly over a local neighbourhood
therefore recovers the structural dip (a, c) with the per-well datums projected
out.

The eval-zone slope then follows from geometry alone:
    dTVT/dMD = a*dX/dMD + c*dY/dMD - dZ/dMD
where the trajectory derivatives are known exactly.
"""
import numpy as np, pandas as pd, glob, os
from scipy.spatial import cKDTree

DATA = os.environ.get('ROGII_DATA', 'data')

_CLOUD = None      # (wid array, X, Y, S) subsampled over all train wells
_TREE = None
_CENT = None       # per-well centroid table


def build_cloud(stride=25, split='train'):
    """Subsample (X, Y, S=TVT+Z) from every train well that has labels."""
    global _CLOUD, _TREE, _CENT
    if _CLOUD is not None:
        return _CLOUD
    wid, X, Y, S = [], [], [], []
    for p in sorted(glob.glob(f'{DATA}/{split}/*__horizontal_well.csv')):
        w = p.split('/')[-1].replace('__horizontal_well.csv', '')
        d = pd.read_csv(p, usecols=['X', 'Y', 'Z', 'TVT'])
        s = d['TVT'].to_numpy(float) + d['Z'].to_numpy(float)
        m = np.isfinite(s) & np.isfinite(d['X']) & np.isfinite(d['Y'])
        if m.sum() < 50:
            continue
        idx = np.flatnonzero(m)[::stride]
        wid.append(np.full(len(idx), w)); X.append(d['X'].to_numpy(float)[idx])
        Y.append(d['Y'].to_numpy(float)[idx]); S.append(s[idx])
    _CLOUD = (np.concatenate(wid), np.concatenate(X), np.concatenate(Y), np.concatenate(S))
    _TREE = cKDTree(np.column_stack(_CLOUD[1:3]))
    df = pd.DataFrame({'w': _CLOUD[0], 'x': _CLOUD[1], 'y': _CLOUD[2]})
    _CENT = df.groupby('w')[['x', 'y']].median()
    return _CLOUD


def local_dip(qx, qy, exclude=None, k_wells=12, min_rows=200, ridge=1e-6):
    """Joint fit S = b_well + a*X + c*Y over the nearest wells. Returns (a, c, n_wells)."""
    build_cloud()
    wid, X, Y, S = _CLOUD
    if _CENT is None or len(_CENT) == 0:
        return 0.0, 0.0, 0
    cent = _CENT if exclude is None else _CENT.drop(index=exclude, errors='ignore')
    d = np.hypot(cent['x'].values - qx, cent['y'].values - qy)
    take = cent.index.values[np.argsort(d)[:k_wells]]
    m = np.isin(wid, take)
    if m.sum() < min_rows:
        return 0.0, 0.0, 0
    xs, ys, ss, ws = X[m], Y[m], S[m], wid[m]
    uw, inv = np.unique(ws, return_inverse=True)
    nw = len(uw)
    # design: [per-well intercepts | X | Y], centred for conditioning
    x0, y0 = xs.mean(), ys.mean()
    A = np.zeros((len(xs), nw + 2))
    A[np.arange(len(xs)), inv] = 1.0
    A[:, nw] = xs - x0
    A[:, nw + 1] = ys - y0
    AT = A.T
    coef = np.linalg.solve(AT @ A + ridge * np.eye(nw + 2), AT @ ss)
    return float(coef[nw]), float(coef[nw + 1]), nw


def surface_shape(hw, wid=None, k_wells=10, k_pts=60, power=2.0, max_dev=40.0,
                  prefix_datum=False, datum_lim=15.0):
    """Predict the whole TVT curve, not just its slope.

    Fitting S = b_i + a*X + c*Y over the neighbourhood gives the per-well datums
    b_i; subtracting them leaves a datum-free cloud of formation elevation E.
    Interpolating E along the eval trajectory and differencing against the anchor
    gives the shape, curvature included:

        TVT(md) = t0 + [E(x,y) - E(x0,y0)] - [Z - Z0]
    """
    build_cloud()
    wid_a, X, Y, S = _CLOUD
    t = hw['TVT_input'].to_numpy(float)
    md = hw['MD'].to_numpy(float); z = hw['Z'].to_numpy(float)
    x = hw['X'].to_numpy(float); y = hw['Y'].to_numpy(float)
    ev = np.flatnonzero(~np.isfinite(t)); kn = np.flatnonzero(np.isfinite(t))
    out = t.copy()
    if len(ev) < 20 or len(kn) < 20:
        if len(kn):
            out[~np.isfinite(out)] = t[kn[-1]]
        return out
    t0, x0, y0, z0 = t[kn[-1]], x[kn[-1]], y[kn[-1]], z[kn[-1]]

    cent = _CENT if wid is None else _CENT.drop(index=wid, errors='ignore')
    d = np.hypot(cent['x'].values - np.median(x[ev]), cent['y'].values - np.median(y[ev]))
    take = cent.index.values[np.argsort(d)[:k_wells]]
    m = np.isin(wid_a, take)
    if m.sum() < 200:
        out[ev] = t0
        return out
    xs, ys, ss, ws = X[m], Y[m], S[m], wid_a[m]
    uw, inv = np.unique(ws, return_inverse=True)
    nw = len(uw)
    cx, cy = xs.mean(), ys.mean()
    A = np.zeros((len(xs), nw + 2))
    A[np.arange(len(xs)), inv] = 1.0
    A[:, nw] = xs - cx; A[:, nw + 1] = ys - cy
    AT = A.T
    coef = np.linalg.solve(AT @ A + 1e-6 * np.eye(nw + 2), AT @ ss)
    # datum-free formation elevation at every cloud point
    E = ss - coef[inv]
    tree = cKDTree(np.column_stack([xs, ys]))

    # Query the surface along the eval path, at the anchor, and across the whole
    # known prefix. b_well = TVT + Z - E is a constant, so estimating it from the
    # entire prefix instead of the single anchor point cuts the datum noise --
    # and the datum is the dominant remaining error term.
    kn_s = kn[::max(1, len(kn) // 400)]
    q = np.column_stack([np.r_[x0, x[ev], x[kn_s]], np.r_[y0, y[ev], y[kn_s]]])
    dd, ii = tree.query(q, k=min(k_pts, len(xs)))
    w = 1.0 / np.maximum(dd, 1.0) ** power
    Eq = (w * E[ii]).sum(axis=1) / w.sum(axis=1)
    n_ev = len(ev)
    E_anchor, E_ev, E_kn = Eq[0], Eq[1:1 + n_ev], Eq[1 + n_ev:]

    if prefix_datum:
        b_hat = float(np.nanmedian(t[kn_s] + z[kn_s] - E_kn))
        pred = b_hat + E_ev - z[ev]
        # never let the prefix-averaged datum run away from the anchor
        anchor_pred = t0 + (E_ev - E_anchor) - (z[ev] - z0)
        pred = anchor_pred + np.clip(pred - anchor_pred, -datum_lim, datum_lim)
    else:
        pred = t0 + (E_ev - E_anchor) - (z[ev] - z0)
    out[ev] = t0 + np.clip(pred - t0, -max_dev, max_dev)
    return out


def predicted_slope(hw, wid=None, k_wells=12):
    """Geometric prediction of d(TVT)/d(MD) over the eval zone."""
    t = hw['TVT_input'].to_numpy(float)
    md = hw['MD'].to_numpy(float); z = hw['Z'].to_numpy(float)
    x = hw['X'].to_numpy(float); y = hw['Y'].to_numpy(float)
    ev = np.flatnonzero(~np.isfinite(t)); kn = np.flatnonzero(np.isfinite(t))
    if len(ev) < 20 or len(kn) < 20:
        return 0.0, 0.0, 0.0, 0
    a, c, nw = local_dip(float(np.median(x[ev])), float(np.median(y[ev])),
                         exclude=wid, k_wells=k_wells)
    dx_ = md[ev] - md[kn[-1]]
    def rate(v):
        vv = v[ev] - v[kn[-1]]
        return float(dx_ @ vv / max(dx_ @ dx_, 1e-9))
    rx, ry, rz = rate(x), rate(y), rate(z)
    return a * rx + c * ry - rz, a, c, nw
