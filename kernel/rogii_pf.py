# ROGII Wellbore Geology Prediction — Particle-Filter TVT tracking
# Self-contained Kaggle kernel. Internet OFF. Reads /kaggle/input, writes submission.csv.
#
# Method (physically grounded geosteering inversion):
#   Track the structural level S = TVT + Z as a momentum random walk along MD.
#   At each step the gamma-ray likelihood (observed GR vs typewell GR at the
#   implied TVT = S - Z) reweights a particle cloud; resample when degenerate.
#   tvt = S - Z. Ensemble over many seeds, weighted by path log-likelihood,
#   tames the stochasticity. Falls back to last-known TVT on any failure.
import os, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
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

N_SEEDS = 96
N_PART = 500
SHRINK = 0.6      # blend PF toward the anchor (validated robust on backtest)
CLIP = 50.0       # bound divergence: |pred - anchor| <= CLIP before shrink

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
    ki = np.where(kn)[0]; ei = np.where(ev)[0]
    last = ki[-1]
    tw_at_k = np.interp(ti[ki], tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(GR[ki]-tw_at_k), 10., 60.))
    tail = ki[-30:]
    dS = np.diff(ti[tail]+Z[tail]); dm = np.diff(MD[tail]); m = dm > 0
    ir = float(np.median(dS[m]/dm[m])) if m.sum() >= 3 else 0.0
    N = n_particles; rng = np.random.default_rng(seed)
    pos = (ti[last]+Z[last]) + init_spread*rng.standard_normal(N)
    rate = ir + 0.01*rng.standard_normal(N)
    w = np.ones(N)/N
    res = np.empty(len(ei)); prev = MD[last]; ll = 0.0
    for i, gi in enumerate(ei):
        dms = max(MD[gi]-prev, 1.0)
        rate = MOM*rate + VN*rng.standard_normal(N)
        pos = pos + rate*dms + PN*rng.standard_normal(N)
        tvt_p = np.clip(pos-Z[gi], tw_tvt[0]-100, tw_tvt[-1]+100)
        pos = tvt_p + Z[gi]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (GR[gi]-eg)/gs
        lk = np.maximum(np.exp(-0.5*np.minimum(d**2, 600.)), 1e-300)
        ll += np.log(max(float((w*lk).sum()), 1e-300))
        w = w*lk; ws = w.sum(); w = w/ws if ws > 0 else np.ones(N)/N
        if 1.0/(w**2).sum() < RESAMP*N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0/N)
            idx = np.clip(np.searchsorted(cum, u0+np.arange(N)/N), 0, N-1)
            pos = pos[idx]+RP*rng.standard_normal(N)
            rate = rate[idx]+RR*rng.standard_normal(N)
            w = np.ones(N)/N
        res[i] = float(np.dot(w, pos-Z[gi]))
        prev = MD[gi]
    out[ei] = res
    return out, ll

def pf_ensemble(hw, tw, n_seeds=96, n_particles=500, scale=5.0):
    preds = []; liks = []
    for s in range(n_seeds):
        p, ll = run_pf(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p); liks.append(ll)
    liks = np.array(liks); wn = np.exp((liks-liks.max())/scale); wn /= wn.sum()
    return (wn[:, None]*np.stack(preds, 0)).sum(0)

def predict_one(wid):
    try:
        hw = pd.read_csv(f'{INP}/test/{wid}__horizontal_well.csv')
        tw = pd.read_csv(f'{INP}/test/{wid}__typewell.csv')
        pred = pf_ensemble(hw, tw, n_seeds=N_SEEDS, n_particles=N_PART)
        ti = hw['TVT_input'].values.astype(float)
        anchor = ti[np.isfinite(ti)][-1] if np.isfinite(ti).any() else float(tw['TVT'].median())
        pred = np.where(np.isfinite(pred), pred, anchor)
        # robustness: clip divergence, then shrink toward the anchor
        pred = anchor + SHRINK*(np.clip(pred, anchor-CLIP, anchor+CLIP) - anchor)
        # keep known zone exactly
        pred = np.where(np.isfinite(ti), ti, pred)
        return wid, pred
    except Exception as e:
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
    if HAVE_JOBLIB and len(wells) > 1:
        results = Parallel(n_jobs=-1, verbose=5)(delayed(predict_one)(wid) for wid in wells)
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
