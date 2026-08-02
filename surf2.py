"""Regional structural surfaces built from the exact formation-top columns.

The horizontal-well CSVs carry ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA: the absolute
depth of each formation top at that (X, Y). Along a train well those columns are
exact -- reconstructing TVT from them scores RMSE 0.007 over the eval zone. They
are stripped from the test wells, but (X, Y, Z) are not, so a surface fitted on
the train cloud can still be evaluated along a test trajectory.

This is the same identity dipfield.py chased (S = b_well + E(X, Y)) with the
confounder removed: formation depths are absolute, so there is no per-well datum
to fit alongside the dip, and every train row is a sample rather than one point
per well.
"""
import numpy as np, pandas as pd, glob, os
from scipy.spatial import cKDTree

DATA = os.environ.get('ROGII_DATA', 'data')
FC = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']


def well_ids(split='train'):
    return sorted(p.split('/')[-1].split('__')[0]
                  for p in glob.glob(f'{DATA}/{split}/*__horizontal_well.csv'))


def build_cloud(wids, stride=25):
    """Subsample every well's trajectory into (x, y, depth per formation)."""
    X, Y, D, W = [], [], [], []
    for w in wids:
        hw = pd.read_csv(f'{DATA}/train/{w}__horizontal_well.csv')
        if not all(c in hw.columns for c in FC):
            continue
        s = hw.iloc[::stride]
        d = s[FC].to_numpy(float)
        m = np.isfinite(d).all(1) & np.isfinite(s['X']) & np.isfinite(s['Y'])
        if m.sum() == 0:
            continue
        X.append(s['X'].to_numpy(float)[m]); Y.append(s['Y'].to_numpy(float)[m])
        D.append(d[m]); W.append(np.full(m.sum(), w))
    return (np.concatenate(X), np.concatenate(Y),
            np.concatenate(D), np.concatenate(W))


class SurfaceModel:
    """Local quadratic fit of each formation surface around a query point."""

    def __init__(self, cloud, exclude=None):
        x, y, d, w = cloud
        if exclude is not None:
            m = ~np.isin(w, exclude)
            x, y, d, w = x[m], y[m], d[m], w[m]
        self.x, self.y, self.d, self.w = x, y, d, w
        self.tree = cKDTree(np.column_stack([x, y]))

    def eval_at(self, qx, qy, k=400, deg=2, ridge=1e-6):
        """Fit one local surface per formation at the centroid of the query,
        then evaluate it along the whole query path. Returns (n, 6)."""
        cx, cy = float(np.mean(qx)), float(np.mean(qy))
        k = min(k, len(self.x))
        _, idx = self.tree.query([cx, cy], k=k)
        idx = np.atleast_1d(idx)
        dx = self.x[idx] - cx; dy = self.y[idx] - cy
        qdx = qx - cx; qdy = qy - cy
        if deg == 1:
            A = np.column_stack([np.ones_like(dx), dx, dy])
            Q = np.column_stack([np.ones_like(qdx), qdx, qdy])
        else:
            A = np.column_stack([np.ones_like(dx), dx, dy,
                                 dx * dx, dx * dy, dy * dy])
            Q = np.column_stack([np.ones_like(qdx), qdx, qdy,
                                 qdx * qdx, qdx * qdy, qdy * qdy])
        sc = np.maximum(np.abs(A).max(0), 1e-9)
        An = A / sc; Qn = Q / sc
        G = An.T @ An + ridge * np.trace(An.T @ An) / An.shape[1] * np.eye(An.shape[1])
        coef = np.linalg.solve(G, An.T @ self.d[idx])          # (p, 6)
        return Qn @ coef

    def dist_to_cloud(self, qx, qy):
        d, _ = self.tree.query(np.column_stack([qx, qy]), k=1)
        return d


def predict_well(sm, hw, k=400, deg=2, shrink=1.0):
    """Anchor on the last known row, then follow the surface. Returns the
    prediction over the eval rows and the surface consensus spread."""
    t = hw['TVT_input'].to_numpy(float)
    z = hw['Z'].to_numpy(float)
    x = hw['X'].to_numpy(float); y = hw['Y'].to_numpy(float)
    kn = np.flatnonzero(np.isfinite(t))
    ev = np.flatnonzero(~np.isfinite(t))
    if len(kn) == 0 or len(ev) == 0:
        return None
    a = kn[-1]
    q = np.concatenate([[a], ev])
    E = sm.eval_at(x[q], y[q], k=k, deg=deg)                  # (1+n_ev, 6)
    dE = E[1:] - E[0]                                          # per formation
    # the six surfaces should move together; their spread is the honest
    # uncertainty of the structural read
    step = dE.mean(1)
    spread = dE.std(1)
    pred = t[a] + shrink * (step - (z[ev] - z[a]))
    return pred, spread, ev, t[a]
