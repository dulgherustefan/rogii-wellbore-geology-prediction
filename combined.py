"""Confidence-weighted blend of three TVT predictors:
  - const     : hold last known TVT (anchor)
  - PF        : particle-filter GR tracking ensemble
  - struct    : neighbour formation-surface (TVT = b_well + E(X,Y) - Z)
Per-well weights from known-zone reconstruction error (pseudo-eval CV for PF,
fit residual for struct). Catches PF wrong-direction divergence with struct."""
import numpy as np, pandas as pd
import pf

FORM = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

def struct_predict(hw, XY, EE, WID, tree, exclude=None, k_other=16, kq=260, ls=200.0):
    ti = hw['TVT_input'].values.astype(float)
    Z = hw['Z'].values.astype(float)
    kn = np.isfinite(ti)
    q = hw[['X', 'Y']].values
    d, ii = tree.query(q, k=kq)
    n = len(q); Ep = np.full((n, 6), np.nan)
    for j in range(n):
        idx = ii[j]; dd = d[j]
        if exclude is not None:
            m = WID[idx] != exclude; idx = idx[m]; dd = dd[m]
        idx = idx[:k_other]; dd = dd[:k_other]
        if len(idx) < 5:
            continue
        wv = np.exp(-(dd/ls)**2)+1e-9
        Ep[j] = np.average(EE[idx], axis=0, weights=wv)
    preds = []
    for f in range(6):
        Ef = Ep[:, f]; m = kn & np.isfinite(Ef)
        if m.sum() < 10:
            continue
        bw = np.median(ti[m]+Z[m]-Ef[m]); preds.append(bw+Ef-Z)
    anchor = ti[kn][-1] if kn.any() else 0.0
    if not preds:
        return np.full(n, anchor), 1e9
    out = np.nanmean(np.array(preds), axis=0)
    out[~np.isfinite(out)] = anchor
    kres = np.sqrt(np.nanmean((out[kn]-ti[kn])**2))
    return out, kres

def pf_pseudo_cv(hw, tw, frac=0.25, n_seeds=24):
    """Hold out last `frac` of known zone, run PF, measure RMSE there."""
    ti = hw['TVT_input'].values.astype(float)
    kn = np.where(np.isfinite(ti))[0]
    if len(kn) < 80:
        return 6.0
    n_hold = int(len(kn)*frac)
    if n_hold < 20:
        return 6.0
    hold = kn[-n_hold:]
    hw2 = hw.copy()
    ti2 = ti.copy(); ti2[hold] = np.nan
    hw2['TVT_input'] = ti2
    try:
        p = pf.pf_ensemble(hw2, tw, n_seeds=n_seeds)
        return float(np.sqrt(np.mean((p[hold]-ti[hold])**2)))
    except Exception:
        return 6.0

def predict(hw, tw, XY, EE, WID, tree, exclude=None,
            n_seeds=48, tau_pf=6.0, tau_st=5.0, pf_shrink=0.85):
    ti = hw['TVT_input'].values.astype(float)
    kn = np.isfinite(ti)
    anchor = ti[kn][-1] if kn.any() else float(tw['TVT'].median())
    n = len(hw)
    # predictors
    pf_pred = pf.pf_ensemble(hw, tw, n_seeds=n_seeds)
    pf_pred = anchor + pf_shrink*(np.clip(pf_pred, anchor-55, anchor+55) - anchor)
    st_pred, st_kres = struct_predict(hw, XY, EE, WID, tree, exclude)
    st_pred = np.clip(st_pred, anchor-90, anchor+90)
    pf_kres = pf_pseudo_cv(hw, tw, n_seeds=max(16, n_seeds//2))
    # weights (const always has a small floor weight)
    w_const = 0.15
    w_pf = np.exp(-(pf_kres/tau_pf)**2)
    w_st = np.exp(-(st_kres/tau_st)**2)
    tot = w_const + w_pf + w_st
    pred = (w_const*anchor + w_pf*pf_pred + w_st*st_pred)/tot
    pred[~np.isfinite(pred)] = anchor
    pred[~kn.any()] = anchor if False else pred[~kn.any()] if kn.any() else anchor
    return pred, dict(pf_kres=pf_kres, st_kres=st_kres, w_pf=float(w_pf), w_st=float(w_st))
