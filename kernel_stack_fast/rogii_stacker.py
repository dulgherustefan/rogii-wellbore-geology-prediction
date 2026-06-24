# ROGII Wellbore Geology Prediction — Stacked geosteering solution
# Self-contained Kaggle kernel. Internet OFF. Reads /kaggle/input, writes submission.csv.
#
# Pipeline (all signals physically grounded; a GBM learns which to trust per well):
#   base predictors of TVT in the eval zone:
#     pf   : particle filter on S=TVT+Z with typewell GR likelihood (seed ensemble)
#     pf2  : same PF but GR likelihood vs the well's OWN pre-PS GR(TVT) template (slide 9)
#     st   : neighbour formation-surface  TVT=b_well+E(X,Y)-Z  (E from train wells)
#   per-well confidences: PF pseudo-CV, struct known-zone residual, formation dispersion,
#     neighbour coverage. Plus GR/trajectory features.
#   LightGBM (GroupKFold-trained on all train wells) blends them -> TVT-anchor.
import os, glob, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import lightgbm as lgb

INP = '/kaggle/input/rogii-wellbore-geology-prediction'
if not os.path.isdir(INP):
    for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction', 'data']:
        if os.path.isdir(c):
            INP = c; break
TRAIN = f'{INP}/train'; TEST = f'{INP}/test'
FORM = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PF_SEEDS = 24
NJOBS = -1

# ---------------- particle filter ----------------
def build_own_ref(ti, GR, tw_tvt, tw_gr, res=0.5, pad=40):
    kn = np.isfinite(ti) & np.isfinite(GR); tk = ti[kn]; gk = GR[kn]
    if len(tk) < 30:
        return tw_tvt, tw_gr
    lo = min(tk.min(), tw_tvt.min())-pad; hi = max(tk.max(), tw_tvt.max())+pad
    grid = np.arange(lo, hi, res); tw_on = np.interp(grid, tw_tvt, tw_gr)
    bins = np.clip(np.round((tk-lo)/res).astype(int), 0, len(grid)-1)
    sv = np.zeros(len(grid)); cnt = np.zeros(len(grid)); np.add.at(sv, bins, gk); np.add.at(cnt, bins, 1)
    ref = np.full(len(grid), np.nan); ref[cnt > 0] = sv[cnt > 0]/cnt[cnt > 0]; have = np.isfinite(ref)
    if have.sum() > 20:
        A = np.vstack([tw_on[have], np.ones(have.sum())]).T; sol, *_ = np.linalg.lstsq(A, ref[have], rcond=None)
        a, b = sol if 0.2 < sol[0] < 5 else (1.0, 0.0); ref[~have] = (a*tw_on+b)[~have]
    else:
        ref[~have] = tw_on[~have]
    ref = pd.Series(ref).interpolate().bfill().ffill().rolling(3, min_periods=1, center=True).mean().values
    return grid, ref

def run_pf(hw, tw_tvt, tw_gr, seed=42, N=500, init_spread=4.5,
           MOM=0.998, VN=0.002, PN=0.005, RP=0.1, RR=0.001, RESAMP=0.5):
    ti = hw['TVT_input'].values.astype(float); kn = np.isfinite(ti); ev = ~kn; out = ti.copy()
    if ev.sum() == 0 or kn.sum() < 3:
        if ev.any(): out[ev] = ti[kn][-1] if kn.any() else float(np.median(tw_tvt))
        return out, 0.0
    Z = hw['Z'].values.astype(float); MD = hw['MD'].values.astype(float)
    GR = hw['GR'].interpolate(limit_direction='both').fillna(np.mean(tw_gr)).values.astype(float)
    ki = np.where(kn)[0]; ei = np.where(ev)[0]; last = ki[-1]
    tw_at_k = np.interp(ti[ki], tw_tvt, tw_gr); gs = float(np.clip(np.nanstd(GR[ki]-tw_at_k), 10., 60.))
    tail = ki[-30:]; dS = np.diff(ti[tail]+Z[tail]); dm = np.diff(MD[tail]); m = dm > 0
    ir = float(np.median(dS[m]/dm[m])) if m.sum() >= 3 else 0.0
    rng = np.random.default_rng(seed); pos = (ti[last]+Z[last])+init_spread*rng.standard_normal(N)
    rate = ir+0.01*rng.standard_normal(N); w = np.ones(N)/N; res = np.empty(len(ei)); prev = MD[last]; ll = 0.0
    for i, gi in enumerate(ei):
        dms = max(MD[gi]-prev, 1.0); rate = MOM*rate+VN*rng.standard_normal(N)
        pos = pos+rate*dms+PN*rng.standard_normal(N)
        tvt_p = np.clip(pos-Z[gi], tw_tvt[0]-100, tw_tvt[-1]+100); pos = tvt_p+Z[gi]
        eg = np.interp(tvt_p, tw_tvt, tw_gr); d = (GR[gi]-eg)/gs
        lk = np.maximum(np.exp(-0.5*np.minimum(d**2, 600.)), 1e-300)
        ll += np.log(max(float((w*lk).sum()), 1e-300)); w = w*lk; ws = w.sum(); w = w/ws if ws > 0 else np.ones(N)/N
        if 1.0/(w**2).sum() < RESAMP*N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0/N); idx = np.clip(np.searchsorted(cum, u0+np.arange(N)/N), 0, N-1)
            pos = pos[idx]+RP*rng.standard_normal(N); rate = rate[idx]+RR*rng.standard_normal(N); w = np.ones(N)/N
        res[i] = float(np.dot(w, pos-Z[gi])); prev = MD[gi]
    out[ei] = res; return out, ll

def pf_ensemble(hw, tw, n_seeds=40, own_ref=False, scale=5.0):
    tw_s = tw.sort_values('TVT'); tt = tw_s['TVT'].values.astype(float); tg = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    if own_ref:
        tt, tg = build_own_ref(hw['TVT_input'].values.astype(float), hw['GR'].values.astype(float), tt, tg)
    preds = []; liks = []
    for s in range(n_seeds):
        p, ll = run_pf(hw, tt, tg, seed=s); preds.append(p); liks.append(ll)
    liks = np.array(liks); wn = np.exp((liks-liks.max())/scale); wn /= wn.sum()
    return (wn[:, None]*np.stack(preds, 0)).sum(0)

def pf_pseudo(hw, tw, n_seeds=20, frac=0.25):
    ti = hw['TVT_input'].values.astype(float); kn = np.where(np.isfinite(ti))[0]
    if len(kn) < 80: return 6.0
    nh = int(len(kn)*frac)
    if nh < 20: return 6.0
    hold = kn[-nh:]; h2 = hw.copy(); t2 = ti.copy(); t2[hold] = np.nan; h2['TVT_input'] = t2
    try:
        p = pf_ensemble(h2, tw, n_seeds=n_seeds); return float(np.sqrt(np.mean((p[hold]-ti[hold])**2)))
    except Exception:
        return 6.0

# ---------------- structural surface ----------------
def build_cloud():
    XY = []; EE = []; WID = []
    for w in sorted(glob.glob(f'{TRAIN}/*__horizontal_well.csv')):
        wid = w.split('/')[-1].split('__')[0]
        try:
            h = pd.read_csv(w, usecols=['X', 'Y']+FORM).iloc[::2]
        except Exception:
            continue
        XY.append(h[['X', 'Y']].values.astype(float)); EE.append(h[FORM].values.astype(float)); WID += [wid]*len(h)
    return np.vstack(XY), np.vstack(EE), np.array(WID), cKDTree(np.vstack(XY))

CL = {}

def ensure_cloud():
    """Build the formation cloud once per process (lazy; survives loky workers)."""
    if 'tree' not in CL:
        XY, EE, WID, tree = build_cloud()
        CL.update(XY=XY, EE=EE, WID=WID, tree=tree)
    return CL

def struct_pred(h, exclude=None, k=25, ls=180.0, kq=900, decay=0.02):
    c = ensure_cloud()
    XY = c['XY']; EE = c['EE']; WID = c['WID']; tree = c['tree']
    ti = h['TVT_input'].values.astype(float); Z = h['Z'].values.astype(float); kn = np.isfinite(ti)
    anchor = ti[kn][-1] if kn.any() else 0.0
    q = h[['X', 'Y']].values.astype(float); d, ii = tree.query(q, k=min(kq, len(XY)))
    Ep = np.full((len(q), 6), np.nan); cov = np.full(len(q), np.inf)
    for j in range(len(q)):
        idx = ii[j]; dd = d[j]
        if exclude is not None:
            m = WID[idx] != exclude; idx = idx[m]; dd = dd[m]
        idx = idx[:k]; dd = dd[:k]
        if len(idx) < 6: continue
        cov[j] = dd[0]; wv = np.exp(-(dd/ls)**2)+1e-9
        for fi in range(6):
            vals = EE[idx, fi]; ok = np.isfinite(vals)
            if ok.sum() >= 5: Ep[j, fi] = np.average(vals[ok], weights=wv[ok])
    preds = []
    for fi in range(6):
        Ef = Ep[:, fi]; m = kn & np.isfinite(Ef)
        if m.sum() < 10: continue
        mi = np.where(m)[0]; wts = np.exp(decay*(np.arange(len(mi))-len(mi)))
        bw = np.average(ti[mi]+Z[mi]-Ef[mi], weights=wts); preds.append(bw+Ef-Z)
    if not preds:
        return np.full(len(h), anchor), cov, 1e9, np.nan
    out = np.nanmean(np.array(preds), axis=0); out[~np.isfinite(out)] = anchor
    kres = float(np.sqrt(np.nanmean((out[kn]-ti[kn])**2)))
    disp = float(np.nanmean(np.nanstd(Ep, axis=1)))
    return out, cov, kres, disp

FEATS = ['pf_dev', 'pf2_dev', 'st_dev', 'disagree', 'disagree2',
         'st_kres', 'pf_cv', 'st_disp', 'cov', 'md_into', 'md_frac',
         'Z_dev', 'incl', 'lat_dist', 'gr', 'grs', 'dgr', 'gr_dev_anchor', 'gr_rs']

def well_feats(wid, split='train'):
    base = TRAIN if split == 'train' else TEST
    h = pd.read_csv(f'{base}/{wid}__horizontal_well.csv'); tw = pd.read_csv(f'{base}/{wid}__typewell.csv')
    ti = h['TVT_input'].values.astype(float); kn = np.isfinite(ti); em = ~kn
    if em.sum() == 0: return None
    cut = int(np.argmax(em)); anchor = ti[cut-1] if cut > 0 else float(tw['TVT'].median())
    MD = h['MD'].values.astype(float); Z = h['Z'].values.astype(float)
    X = h['X'].values.astype(float); Y = h['Y'].values.astype(float)
    GR = h['GR'].values.astype(float); grf = np.where(np.isfinite(GR), GR, np.nanmedian(GR))
    tws = tw.sort_values('TVT'); twt = tws['TVT'].values; twg = tws['GR'].fillna(tw['GR'].mean()).values
    pf_pred = pf_ensemble(h, tw, n_seeds=PF_SEEDS)
    pf2_pred = pf_ensemble(h, tw, n_seeds=PF_SEEDS, own_ref=True)
    st_pred, cov, st_kres, disp = struct_pred(h, exclude=wid if split == 'train' else None)
    pf_cv = pf_pseudo(h, tw)
    grs = pd.Series(grf).rolling(11, min_periods=1, center=True).mean().values
    dgr = np.gradient(grs, MD); gr_rs = pd.Series(grf).rolling(31, min_periods=1).std().fillna(0).values
    dZ = np.gradient(Z, MD); gr_at_anchor = np.interp(anchor, twt, twg); ev = np.where(em)[0]
    f = pd.DataFrame({
        'wid': wid, 'row': ev, 'anchor': anchor,
        'pf_dev': pf_pred[ev]-anchor, 'pf2_dev': pf2_pred[ev]-anchor, 'st_dev': st_pred[ev]-anchor,
        'disagree': pf_pred[ev]-st_pred[ev], 'disagree2': pf_pred[ev]-pf2_pred[ev],
        'st_kres': st_kres, 'pf_cv': pf_cv, 'st_disp': disp,
        'cov': np.where(np.isfinite(cov[ev]), cov[ev], 9999.0),
        'md_into': MD[ev]-MD[cut], 'md_frac': (MD[ev]-MD[cut])/max(1.0, MD[-1]-MD[cut]),
        'Z_dev': Z[ev]-Z[cut-1], 'incl': dZ[ev], 'lat_dist': np.hypot(X[ev]-X[cut-1], Y[ev]-Y[cut-1]),
        'gr': grf[ev], 'grs': grs[ev], 'dgr': dgr[ev], 'gr_dev_anchor': grf[ev]-gr_at_anchor, 'gr_rs': gr_rs[ev],
    })
    if split == 'train':
        f['y'] = h['TVT'].values[ev]-anchor
    return f

def main():
    t0 = time.time()
    print('building cloud...', flush=True)
    ensure_cloud()
    print(f'cloud {len(CL["XY"])} pts ({time.time()-t0:.0f}s)', flush=True)
    train_wids = [w.split('/')[-1].split('__')[0] for w in sorted(glob.glob(f'{TRAIN}/*__horizontal_well.csv'))]
    print(f'generating train features ({len(train_wids)} wells)...', flush=True)
    tr = [x for x in Parallel(n_jobs=NJOBS, verbose=5)(delayed(well_feats)(w, 'train') for w in train_wids) if x is not None]
    df = pd.concat(tr, ignore_index=True)
    print(f'train feats {df.shape} ({time.time()-t0:.0f}s); training GBM...', flush=True)
    Xtr = df[FEATS].values; ytr = df['y'].values
    params = dict(n_estimators=900, learning_rate=0.03, num_leaves=127, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                  reg_alpha=0.1, objective='regression', verbose=-1, n_jobs=-1)
    models = [lgb.LGBMRegressor(**{**params, 'random_state': s}).fit(Xtr, ytr) for s in (42, 7)]
    sub = pd.read_csv(f'{INP}/sample_submission.csv')
    sub['well'] = sub['id'].str.split('_').str[0]; sub['idx'] = sub['id'].str.split('_').str[1].astype(int)
    test_wids = sorted(sub['well'].unique())
    print(f'predicting {len(test_wids)} test wells...', flush=True)
    te = [x for x in Parallel(n_jobs=NJOBS, verbose=5)(delayed(well_feats)(w, 'test') for w in test_wids) if x is not None]
    out = {}
    for fte in te:
        wid = fte['wid'].iloc[0]; anchor = fte['anchor'].iloc[0]
        pr = np.mean([m.predict(fte[FEATS].values) for m in models], axis=0)
        pr = np.clip(pr, -90, 90)
        # light per-well smoothing
        pr = pd.Series(pr).rolling(15, min_periods=1, center=True).mean().values
        tvt = anchor + pr
        for r, idx in zip(tvt, fte['row'].values):
            out[f'{wid}_{idx}'] = r
    sub['tvt'] = sub['id'].map(out)
    sub['tvt'] = sub['tvt'].ffill().fillna(sub['tvt'].median())
    sub[['id', 'tvt']].to_csv('submission.csv', index=False)
    print(f'wrote submission.csv {len(sub)} rows; NaNs={sub.tvt.isna().sum()} ({time.time()-t0:.0f}s)', flush=True)

if __name__ == '__main__':
    main()
