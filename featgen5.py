"""Generate per-eval-row features for the GBM stacker, for every train well.
Base signals: PF (GR particle filter) + struct (neighbour formation surface) +
GR/trajectory features + per-well confidences. Caches to features.parquet."""
import pandas as pd, numpy as np, glob, time, warnings, sys
warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import pf

FORM = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
DATA = 'data/train'
STEP = 2

# ---- build global formation cloud once ----
_wells = sorted(glob.glob(f'{DATA}/*__horizontal_well.csv'))
_XY = []; _EE = []; _WID = []
for w in _wells:
    wid = w.split('/')[-1].split('__')[0]
    h = pd.read_csv(w, usecols=['X', 'Y']+FORM).iloc[::STEP]
    _XY.append(h[['X', 'Y']].values.astype(float)); _EE.append(h[FORM].values.astype(float)); _WID += [wid]*len(h)
BIGXY = np.vstack(_XY); BIGEE = np.vstack(_EE); BIGWID = np.array(_WID)
TREE = cKDTree(BIGXY)
ALLWIDS = [w.split('/')[-1].split('__')[0] for w in _wells]

# per-well CENTROID cloud (one median point per well) for plane fitting — avoids
# the collinearity blow-up from dense same-well lateral points.
_CWID = np.array(ALLWIDS)
_CXY = np.zeros((len(ALLWIDS), 2)); _CEE = np.full((len(ALLWIDS), 6), np.nan)
for _i, _w in enumerate(ALLWIDS):
    _m = BIGWID == _w
    _CXY[_i] = np.nanmedian(BIGXY[_m], axis=0)
    _CEE[_i] = np.nanmedian(BIGEE[_m], axis=0)
_CSCALE = np.where(_CXY.std(0) < 1e-3, 1.0, _CXY.std(0))
CTREE = cKDTree(_CXY/_CSCALE)

def struct_E(qxy, exclude, k=25, ls=180.0, kq=900):
    d, ii = TREE.query(qxy, k=min(kq, len(BIGXY)))
    Ep = np.full((len(qxy), 6), np.nan); cov = np.full(len(qxy), np.inf)
    for j in range(len(qxy)):
        idx = ii[j]; dd = d[j]
        if exclude is not None:
            m = BIGWID[idx] != exclude; idx = idx[m]; dd = dd[m]
        idx = idx[:k]; dd = dd[:k]
        if len(idx) < 6:
            continue
        cov[j] = dd[0]; wv = np.exp(-(dd/ls)**2)+1e-9
        for fi in range(6):
            vals = BIGEE[idx, fi]; ok = np.isfinite(vals)
            if ok.sum() >= 5:
                Ep[j, fi] = np.average(vals[ok], weights=wv[ok])
    return Ep, cov

def struct_E_plane(qxy, exclude, k=10):
    """Local weighted PLANE fit per formation over the k nearest WELL CENTROIDS
    (one point per well -> no same-well collinearity). E ~ a*X + b*Y + c, predicted
    at the query XY. Captures dip/trend; better than IDW where the surface is sloped."""
    q = qxy/_CSCALE
    nf = min(k+6, len(_CWID))
    d, ii = CTREE.query(q, k=nf)
    Ep = np.full((len(qxy), 6), np.nan)
    for j in range(len(qxy)):
        idx = ii[j]; dd = d[j]
        if exclude is not None:
            m = _CWID[idx] != exclude; idx = idx[m]; dd = dd[m]
        idx = idx[:k]; dd = dd[:k]
        if len(idx) < 5:
            continue
        w = 1.0/(dd+1e-3)
        Xn = _CXY[idx, 0]; Yn = _CXY[idx, 1]
        qx, qy = qxy[j]
        for fi in range(6):
            vals = _CEE[idx, fi]; ok = np.isfinite(vals)
            if ok.sum() < 5:
                continue
            xx = Xn[ok]; yy = Yn[ok]; zz = vals[ok]; ww = w[ok]
            # center X,Y for conditioning
            mx = xx.mean(); my = yy.mean()
            A = np.column_stack([xx-mx, yy-my, np.ones(len(xx))])
            AT = A.T*ww
            try:
                coef = np.linalg.solve(AT@A + 1e-3*np.eye(3), AT@zz)
                Ep[j, fi] = coef[0]*(qx-mx) + coef[1]*(qy-my) + coef[2]
            except Exception:
                Ep[j, fi] = np.average(zz, weights=ww)
    return Ep

def struct_pred(h, exclude, decay=0.02):
    ti = h['TVT_input'].values.astype(float); Z = h['Z'].values.astype(float); kn = np.isfinite(ti)
    anchor = ti[kn][-1] if kn.any() else 0.0
    Ep, cov = struct_E(h[['X', 'Y']].values.astype(float), exclude)
    preds = []
    for fi in range(6):
        Ef = Ep[:, fi]; m = kn & np.isfinite(Ef)
        if m.sum() < 10:
            continue
        mi = np.where(m)[0]; wts = np.exp(decay*(np.arange(len(mi))-len(mi)))
        bw = np.average(ti[mi]+Z[mi]-Ef[mi], weights=wts); preds.append(bw+Ef-Z)
    if not preds:
        nanrow = np.full(len(h), anchor)
        return np.full(len(h), anchor), cov, 1e9, {}, np.nan
    arr = np.array(preds)
    out = np.nanmean(arr, axis=0); out[~np.isfinite(out)] = anchor
    kres = float(np.sqrt(np.nanmean((out[kn]-ti[kn])**2)))
    # per-formation predictions (dict fi->array) and local E-dispersion (fault proxy)
    perf = {}
    fii = 0
    for fi in range(6):
        Ef = Ep[:, fi]; m = kn & np.isfinite(Ef)
        if m.sum() < 10:
            continue
        mi = np.where(m)[0]; wts = np.exp(decay*(np.arange(len(mi))-len(mi)))
        bw = np.average(ti[mi]+Z[mi]-Ef[mi], weights=wts); perf[fi] = bw+Ef-Z
    disp = float(np.nanmean(np.nanstd(Ep, axis=1)))  # avg spread across formations
    return out, cov, kres, perf, disp

def struct_plane_pred(h, exclude, decay=0.02):
    """TVT prediction from the local plane-fit formation surface (datum via recent-
    weighted b_well per formation, averaged). Returns (pred, known-zone rmse)."""
    ti = h['TVT_input'].values.astype(float); Z = h['Z'].values.astype(float); kn = np.isfinite(ti)
    anchor = ti[kn][-1] if kn.any() else 0.0
    Ep = struct_E_plane(h[['X', 'Y']].values.astype(float), exclude)
    preds = []
    for fi in range(6):
        Ef = Ep[:, fi]; m = kn & np.isfinite(Ef)
        if m.sum() < 10:
            continue
        mi = np.where(m)[0]; wts = np.exp(decay*(np.arange(len(mi))-len(mi)))
        bw = np.average(ti[mi]+Z[mi]-Ef[mi], weights=wts); preds.append(bw+Ef-Z)
    if not preds:
        return np.full(len(h), anchor), 1e9
    out = np.nanmean(np.array(preds), axis=0); out[~np.isfinite(out)] = anchor
    kres = float(np.sqrt(np.nanmean((out[kn]-ti[kn])**2)))
    return out, kres

def roll(a, k):
    s = pd.Series(a)
    return s.rolling(k, min_periods=1, center=True).mean().values, s.rolling(k, min_periods=1).std().fillna(0).values

def multi_scale_ncc(ti, GR, anchor, band=45.0, hws=(8, 15, 25), stride=2):
    """Slide-9 signal: correlate horizontal GR windows against the GR(TVT) template
    from the KNOWN zone (own high-res log), restricted to template positions whose
    TVT is within +/-band of the anchor (continuity prior). Returns per-row TVT
    estimate + NCC score for each window size."""
    kn = np.isfinite(ti)
    ktvt = ti[kn]; kgr = GR[kn]
    n = len(GR)
    fb = np.nanmedian(GR)
    hg_full = pd.Series(np.where(np.isfinite(GR), GR, fb)).rolling(5, center=True, min_periods=1).mean().values
    out = []
    order = np.argsort(ktvt)
    kt = ktvt[order]
    kg = pd.Series(np.where(np.isfinite(kgr[order]), kgr[order], fb)).rolling(5, center=True, min_periods=1).mean().values
    for hw in hws:
        win = 2*hw+1; nk = len(kt)
        default = np.full(n, anchor, np.float32)
        if nk < win+1:
            out.append((default, np.zeros(n, np.float32))); continue
        sts = np.arange(0, nk-win+1, stride)
        ctrtvt = kt[np.clip(sts+hw, 0, nk-1)]
        keep = np.abs(ctrtvt - anchor) <= band
        sts = sts[keep]; ctrtvt = ctrtvt[keep]
        if len(sts) < 2:
            out.append((default, np.zeros(n, np.float32))); continue
        C = kg[sts[:, None]+np.arange(win)[None, :]]
        Cn = (C-C.mean(1, keepdims=True))/(C.std(1, keepdims=True)+1e-6)
        hp = np.pad(hg_full, hw, mode='edge')
        H = hp[np.arange(n)[:, None]+np.arange(win)[None, :]]
        Hn = (H-H.mean(1, keepdims=True))/(H.std(1, keepdims=True)+1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1); score = ncc.max(1)
        out.append((ctrtvt[best].astype(np.float32), score.astype(np.float32)))
    return out

def well_feats(wid, is_train=True, pf_seeds=40):
    h = pd.read_csv(f'{DATA}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{DATA}/{wid}__typewell.csv')
    ti = h['TVT_input'].values.astype(float); kn = np.isfinite(ti); em = ~kn
    if em.sum() == 0:
        return None
    cut = int(np.argmax(em)); anchor = ti[cut-1] if cut > 0 else float(tw['TVT'].median())
    MD = h['MD'].values.astype(float); Z = h['Z'].values.astype(float)
    X = h['X'].values.astype(float); Y = h['Y'].values.astype(float)
    GR = h['GR'].values.astype(float)
    grf = np.where(np.isfinite(GR), GR, np.nanmedian(GR))
    twt = tw.sort_values('TVT')['TVT'].values; twg = tw.sort_values('TVT')['GR'].fillna(tw['GR'].mean()).values
    # base predictors
    pf_pred = pf.pf_ensemble(h, tw, n_seeds=pf_seeds)
    pf2_pred = pf.pf_ensemble(h, tw, n_seeds=pf_seeds, own_ref=True)
    # pf3: more responsive PF (higher process noise, lower momentum) -> tracks GR faster
    pf3_pred = pf.pf_ensemble(h, tw, n_seeds=pf_seeds, MOM=0.99, VN=0.006, PN=0.012)
    st_pred, cov, st_kres, perf, disp = struct_pred(h, exclude=wid if is_train else None)
    st_plane_pred, st_plane_kres = struct_plane_pred(h, exclude=wid if is_train else None)
    pf_cv = pf_pseudo(h, tw, pf_seeds//2)
    # slide-9 NCC: align horizontal GR windows to own known-zone GR(TVT) template
    ncc = multi_scale_ncc(ti, grf, anchor, band=45.0, hws=(8, 15, 25))
    ncc8_e, ncc8_s = ncc[0]; ncc15_e, ncc15_s = ncc[1]; ncc25_e, ncc25_s = ncc[2]
    ncc_cons = (ncc8_e*np.maximum(ncc8_s, 0) + ncc15_e*np.maximum(ncc15_s, 0) +
                ncc25_e*np.maximum(ncc25_s, 0)) / \
               (np.maximum(ncc8_s, 0)+np.maximum(ncc15_s, 0)+np.maximum(ncc25_s, 0)+1e-6)
    ncc_sc = (ncc8_s + ncc15_s + ncc25_s) / 3.0
    # GR derived
    grs = pd.Series(grf).rolling(11, min_periods=1, center=True).mean().values
    dgr = np.gradient(grs, MD)
    gr_rm, gr_rs = roll(grf, 31)
    dZ = np.gradient(Z, MD)
    gr_at_anchor = np.interp(anchor, twt, twg)
    ev = np.where(em)[0]
    f = pd.DataFrame({
        'wid': wid, 'row': ev, 'anchor': anchor,
        'pf_dev': pf_pred[ev]-anchor,
        'pf2_dev': pf2_pred[ev]-anchor,
        'st_dev': st_pred[ev]-anchor,
        'disagree': pf_pred[ev]-st_pred[ev],
        'disagree2': pf_pred[ev]-pf2_pred[ev],
        'st_kres': st_kres, 'pf_cv': pf_cv, 'st_disp': disp,
        'cov': np.where(np.isfinite(cov[ev]), cov[ev], 9999.0),
        'md_into': MD[ev]-MD[cut], 'md_frac': (MD[ev]-MD[cut])/max(1.0, MD[-1]-MD[cut]),
        'Z_dev': Z[ev]-Z[cut-1], 'incl': dZ[ev],
        'lat_dist': np.hypot(X[ev]-X[cut-1], Y[ev]-Y[cut-1]),
        'gr': grf[ev], 'grs': grs[ev], 'dgr': dgr[ev],
        'gr_dev_anchor': grf[ev]-gr_at_anchor, 'gr_rs': gr_rs[ev],
        'ncc_cons_dev': ncc_cons[ev]-anchor, 'ncc_sc': ncc_sc[ev],
        'ncc8_dev': ncc8_e[ev]-anchor, 'ncc8_sc': ncc8_s[ev],
        'ncc25_dev': ncc25_e[ev]-anchor, 'ncc25_sc': ncc25_s[ev],
        'ncc_dis': ncc_cons[ev]-pf_pred[ev],
        'pf3_dev': pf3_pred[ev]-anchor, 'disagree3': pf_pred[ev]-pf3_pred[ev],
        # plane-fit structural surface (captures dip; better than IDW across faults)
        'stp_dev': st_plane_pred[ev]-anchor, 'stp_kres': st_plane_kres,
        'stp_dis': st_plane_pred[ev]-st_pred[ev],
        'stp_pf_dis': st_plane_pred[ev]-pf_pred[ev],
        # GR-offset features: gr minus typewell GR sampled at anchor +/- offset
        # (lets the GBM read the local GR signature around the held level)
        'tda_m20': grf[ev]-np.interp(anchor-20, twt, twg),
        'tda_m8': grf[ev]-np.interp(anchor-8, twt, twg),
        'tda_0': grf[ev]-np.interp(anchor, twt, twg),
        'tda_p8': grf[ev]-np.interp(anchor+8, twt, twg),
        'tda_p20': grf[ev]-np.interp(anchor+20, twt, twg),
    })
    if is_train:
        f['y'] = h['TVT'].values[ev] - anchor
    return f

def pf_pseudo(h, tw, n_seeds, frac=0.25):
    ti = h['TVT_input'].values.astype(float); kn = np.where(np.isfinite(ti))[0]
    if len(kn) < 80:
        return 6.0
    nh = int(len(kn)*frac)
    if nh < 20:
        return 6.0
    hold = kn[-nh:]; h2 = h.copy(); t2 = ti.copy(); t2[hold] = np.nan; h2['TVT_input'] = t2
    try:
        p = pf.pf_ensemble(h2, tw, n_seeds=n_seeds)
        return float(np.sqrt(np.mean((p[hold]-ti[hold])**2)))
    except Exception:
        return 6.0

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(ALLWIDS)
    wl = ALLWIDS[:n]
    t0 = time.time()
    outs = Parallel(n_jobs=-1, verbose=5)(delayed(well_feats)(wid) for wid in wl)
    outs = [o for o in outs if o is not None]
    df = pd.concat(outs, ignore_index=True)
    df.to_pickle('features5.pkl')
    print(f"{len(df)} rows from {df['wid'].nunique()} wells in {time.time()-t0:.0f}s -> features5.pkl")
