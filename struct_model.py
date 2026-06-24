"""Structural TVT model: TVT = b_well + E_formation(X,Y) - Z.
E_formation predicted from neighbouring train wells' formation picks via
distance-weighted ridge plane (captures local elevation + dip robustly)."""
import numpy as np, pandas as pd, glob, warnings
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

FORM = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

def load_cloud(train_glob='data/train/*__horizontal_well.csv', step=3):
    XY=[]; EE=[]; WID=[]
    for w in sorted(glob.glob(train_glob)):
        wid = w.split('/')[-1].split('__')[0]
        h = pd.read_csv(w, usecols=['X','Y']+FORM).iloc[::step]
        XY.append(h[['X','Y']].values); EE.append(h[FORM].values); WID += [wid]*len(h)
    return np.vstack(XY), np.vstack(EE), np.array(WID)

def predict_E(qxy, XY, EE, tree, WID=None, exclude=None,
              k_other=18, kquery=260, ls=250.0, ridge=1e-4):
    """Distance-weighted ridge plane per formation. Returns (n,6) E and (n,) coverage dist."""
    kq = kquery if (WID is not None and exclude is not None) else k_other
    d, ii = tree.query(qxy, k=kq)
    n = len(qxy)
    out = np.full((n, 6), np.nan)
    cov = np.full(n, np.inf)
    for j in range(n):
        idx = ii[j]; dd = d[j]
        if WID is not None and exclude is not None:
            m = WID[idx] != exclude
            idx = idx[m]; dd = dd[m]
        idx = idx[:k_other]; dd = dd[:k_other]
        if len(idx) < 6:
            continue
        cov[j] = dd[0]
        dx = XY[idx,0]-qxy[j,0]; dy = XY[idx,1]-qxy[j,1]
        wv = np.exp(-(dd/ls)**2)
        A = np.column_stack([np.ones(len(idx)), dx, dy])
        W = wv[:,None]
        AtA = A.T @ (W*A) + ridge*np.diag([0.0,1.0,1.0])*np.sum(wv)
        for f in range(6):
            Aty = A.T @ (wv*EE[idx,f])
            try:
                c = np.linalg.solve(AtA, Aty); out[j,f] = c[0]
            except Exception:
                out[j,f] = np.average(EE[idx,f], weights=wv)
    return out, cov

def predict_well(h, XY, EE, tree, WID=None, exclude=None, b_decay=0.0, **kw):
    ti = h['TVT_input'].values.astype(float)
    em = ~np.isfinite(ti)
    Z = h['Z'].values
    anchor = ti[np.isfinite(ti)][-1] if np.isfinite(ti).any() else 0.0
    Epred, cov = predict_E(h[['X','Y']].values, XY, EE, tree, WID, exclude, **kw)
    kn = np.isfinite(ti)
    preds = []
    for f in range(6):
        Ef = Epred[:,f]; m = kn & np.isfinite(Ef)
        if m.sum() < 10:
            continue
        if b_decay > 0:
            idxk = np.where(m)[0]
            wts = np.exp(b_decay*(np.arange(len(idxk))-len(idxk)))
            bw = np.average((ti[m]+Z[m]-Ef[m]), weights=wts)
        else:
            bw = np.median(ti[m]+Z[m]-Ef[m])
        preds.append(bw + Ef - Z)
    if not preds:
        out = np.full(len(h), anchor)
    else:
        out = np.nanmean(np.array(preds), axis=0)
        out[~np.isfinite(out)] = anchor
    return out, cov, anchor
