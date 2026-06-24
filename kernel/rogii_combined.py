# ROGII Wellbore Geology Prediction — Combined geosteering solution
# Self-contained Kaggle kernel. Internet OFF. Reads /kaggle/input, writes submission.csv.
#
# Three physically-grounded predictors, blended per-well by known-zone reliability:
#   const  : hold last known TVT (anchor) — strong prior, TVT is steered to stay in zone
#   PF     : particle filter tracking structural level S=TVT+Z via gamma-ray likelihood
#            vs the typewell; tvt=S-Z; likelihood-weighted multi-seed ensemble
#   struct : neighbour formation-surface; TVT = b_well + E(X,Y) - Z, where E(X,Y) is
#            interpolated from nearby TRAIN wells' formation picks (common elevation datum)
# Weights from per-well known-zone reconstruction error (pseudo-eval CV for PF,
# fit residual for struct). struct fixes the PF gamma-ray direction ambiguity; const
# is the safe floor. Falls back to anchor on any failure.
import os, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False

INP = '/kaggle/input/rogii-wellbore-geology-prediction'
if not os.path.isdir(INP):
    for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction', 'data']:
        if os.path.isdir(c):
            INP = c; break

FORM = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
N_SEEDS = 48
PF_SHRINK = 0.85

# ---------------- particle filter ----------------
def run_pf(hw, tw, n_particles=500, seed=42, init_spread=4.5,
           MOM=0.998, VN=0.002, PN=0.005, RP=0.1, RR=0.001, RESAMP=0.5):
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    ti = hw['TVT_input'].values.astype(float)
    kn = np.isfinite(ti); ev = ~kn
    out = ti.copy()
    if ev.sum() == 0 or kn.sum() < 3:
        if ev.any():
            out[ev] = ti[kn][-1] if kn.any() else float(np.median(tw_tvt))
        return out, 0.0
    Z = hw['Z'].values.astype(float); MD = hw['MD'].values.astype(float)
    GR = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    ki = np.where(kn)[0]; ei = np.where(ev)[0]; last = ki[-1]
    tw_at_k = np.interp(ti[ki], tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(GR[ki]-tw_at_k), 10., 60.))
    tail = ki[-30:]; dS = np.diff(ti[tail]+Z[tail]); dm = np.diff(MD[tail]); m = dm > 0
    ir = float(np.median(dS[m]/dm[m])) if m.sum() >= 3 else 0.0
    N = n_particles; rng = np.random.default_rng(seed)
    pos = (ti[last]+Z[last]) + init_spread*rng.standard_normal(N)
    rate = ir + 0.01*rng.standard_normal(N); w = np.ones(N)/N
    res = np.empty(len(ei)); prev = MD[last]; ll = 0.0
    for i, gi in enumerate(ei):
        dms = max(MD[gi]-prev, 1.0)
        rate = MOM*rate + VN*rng.standard_normal(N)
        pos = pos + rate*dms + PN*rng.standard_normal(N)
        tvt_p = np.clip(pos-Z[gi], tw_tvt[0]-100, tw_tvt[-1]+100); pos = tvt_p + Z[gi]
        eg = np.interp(tvt_p, tw_tvt, tw_gr); d = (GR[gi]-eg)/gs
        lk = np.maximum(np.exp(-0.5*np.minimum(d**2, 600.)), 1e-300)
        ll += np.log(max(float((w*lk).sum()), 1e-300))
        w = w*lk; ws = w.sum(); w = w/ws if ws > 0 else np.ones(N)/N
        if 1.0/(w**2).sum() < RESAMP*N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0/N)
            idx = np.clip(np.searchsorted(cum, u0+np.arange(N)/N), 0, N-1)
            pos = pos[idx]+RP*rng.standard_normal(N); rate = rate[idx]+RR*rng.standard_normal(N)
            w = np.ones(N)/N
        res[i] = float(np.dot(w, pos-Z[gi])); prev = MD[gi]
    out[ei] = res
    return out, ll

def pf_ensemble(hw, tw, n_seeds=48, n_particles=500, scale=5.0):
    preds = []; liks = []
    for s in range(n_seeds):
        p, ll = run_pf(hw, tw, n_particles=n_particles, seed=s); preds.append(p); liks.append(ll)
    liks = np.array(liks); wn = np.exp((liks-liks.max())/scale); wn /= wn.sum()
    return (wn[:, None]*np.stack(preds, 0)).sum(0)

def pf_pseudo_cv(hw, tw, frac=0.25, n_seeds=24):
    ti = hw['TVT_input'].values.astype(float); kn = np.where(np.isfinite(ti))[0]
    if len(kn) < 80: return 6.0
    n_hold = int(len(kn)*frac)
    if n_hold < 20: return 6.0
    hold = kn[-n_hold:]; hw2 = hw.copy(); ti2 = ti.copy(); ti2[hold] = np.nan; hw2['TVT_input'] = ti2
    try:
        p = pf_ensemble(hw2, tw, n_seeds=n_seeds)
        return float(np.sqrt(np.mean((p[hold]-ti[hold])**2)))
    except Exception:
        return 6.0

# ---------------- structural neighbour surface ----------------
def build_cloud():
    XY = []; EE = []; WID = []
    for w in sorted(glob.glob(f'{INP}/train/*__horizontal_well.csv')):
        wid = w.split('/')[-1].split('__')[0]
        try:
            h = pd.read_csv(w, usecols=['X', 'Y']+FORM).iloc[::3]
        except Exception:
            continue
        XY.append(h[['X', 'Y']].values); EE.append(h[FORM].values); WID += [wid]*len(h)
    return np.vstack(XY), np.vstack(EE), np.array(WID)

def struct_predict(hw, XY, EE, WID, tree, exclude=None, k_other=16, kq=260, ls=200.0):
    ti = hw['TVT_input'].values.astype(float); Z = hw['Z'].values.astype(float); kn = np.isfinite(ti)
    q = hw[['X', 'Y']].values; d, ii = tree.query(q, k=min(kq, len(XY)))
    n = len(q); Ep = np.full((n, 6), np.nan)
    for j in range(n):
        idx = ii[j]; dd = d[j]
        if exclude is not None:
            m = WID[idx] != exclude; idx = idx[m]; dd = dd[m]
        idx = idx[:k_other]; dd = dd[:k_other]
        if len(idx) < 5: continue
        wv = np.exp(-(dd/ls)**2)+1e-9; Ep[j] = np.average(EE[idx], axis=0, weights=wv)
    preds = []
    for f in range(6):
        Ef = Ep[:, f]; m = kn & np.isfinite(Ef)
        if m.sum() < 10: continue
        bw = np.median(ti[m]+Z[m]-Ef[m]); preds.append(bw+Ef-Z)
    anchor = ti[kn][-1] if kn.any() else 0.0
    if not preds: return np.full(n, anchor), 1e9
    out = np.nanmean(np.array(preds), axis=0); out[~np.isfinite(out)] = anchor
    kres = float(np.sqrt(np.nanmean((out[kn]-ti[kn])**2)))
    return out, kres

CLOUD = {}

def predict_one(wid):
    try:
        hw = pd.read_csv(f'{INP}/test/{wid}__horizontal_well.csv')
        tw = pd.read_csv(f'{INP}/test/{wid}__typewell.csv')
        ti = hw['TVT_input'].values.astype(float); kn = np.isfinite(ti)
        anchor = ti[kn][-1] if kn.any() else float(tw['TVT'].median())
        pf_pred = pf_ensemble(hw, tw, n_seeds=N_SEEDS)
        pf_pred = anchor + PF_SHRINK*(np.clip(pf_pred, anchor-55, anchor+55) - anchor)
        st_pred, st_kres = struct_predict(hw, CLOUD['XY'], CLOUD['EE'], CLOUD['WID'], CLOUD['tree'], exclude=None)
        st_pred = np.clip(st_pred, anchor-90, anchor+90)
        pf_kres = pf_pseudo_cv(hw, tw, n_seeds=24)
        w_const = 0.15; w_pf = np.exp(-(pf_kres/6.0)**2); w_st = np.exp(-(st_kres/5.0)**2)
        tot = w_const + w_pf + w_st
        pred = (w_const*anchor + w_pf*pf_pred + w_st*st_pred)/tot
        pred = np.where(np.isfinite(pred), pred, anchor)
        return wid, pred
    except Exception:
        hw = pd.read_csv(f'{INP}/test/{wid}__horizontal_well.csv')
        ti = hw['TVT_input'].values.astype(float)
        anchor = ti[np.isfinite(ti)][-1] if np.isfinite(ti).any() else 0.0
        return wid, np.where(np.isfinite(ti), ti, anchor)

def main():
    sub = pd.read_csv(f'{INP}/sample_submission.csv')
    sub['well'] = sub['id'].str.split('_').str[0]
    sub['idx'] = sub['id'].str.split('_').str[1].astype(int)
    wells = sorted(sub['well'].unique())
    print(f'{len(wells)} test wells; INP={INP}', flush=True)
    print('building structural cloud...', flush=True)
    XY, EE, WID = build_cloud()
    CLOUD['XY'] = XY; CLOUD['EE'] = EE; CLOUD['WID'] = WID; CLOUD['tree'] = cKDTree(XY)
    print(f'cloud {len(XY)} pts', flush=True)
    if HAVE_JOBLIB and len(wells) > 1:
        # threading backend: workers share the in-memory CLOUD (KDTree + arrays)
        results = Parallel(n_jobs=-1, backend='threading', verbose=5)(
            delayed(predict_one)(wid) for wid in wells)
    else:
        results = [predict_one(wid) for wid in wells]
    out = {}
    for wid, pred in results:
        g = sub[sub['well'] == wid]
        for idx in g['idx'].values:
            out[f'{wid}_{idx}'] = pred[idx] if idx < len(pred) else np.nan
    sub['tvt'] = sub['id'].map(out)
    sub['tvt'] = sub['tvt'].ffill().fillna(sub['tvt'].median())
    sub[['id', 'tvt']].to_csv('submission.csv', index=False)
    print('wrote submission.csv', len(sub), 'rows; NaNs:', sub['tvt'].isna().sum(), flush=True)

if __name__ == '__main__':
    main()
