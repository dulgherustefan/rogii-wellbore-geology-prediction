import pandas as pd, numpy as np, glob, warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
import pf
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def pooled(l): return np.sqrt(sum(r**2*n for r,n in l)/sum(n for _,n in l))

def eval_well(w):
    wid=w.split('/')[-1].split('__')[0]
    h=pd.read_csv(w); tw=pd.read_csv(f'data/train/{wid}__typewell.csv')
    ti=h['TVT_input'].values.astype(float); em=~np.isfinite(ti)
    if em.sum()==0: return None
    y=h['TVT'].values; anchor=ti[np.isfinite(ti)][-1]
    pe=pf.pf_ensemble(h,tw,n_seeds=48)
    yt=y[em]; pev=pe[em]
    variants={}
    variants['const']=np.full(em.sum(),anchor)
    variants['pf']=pev
    for cl in [40,55]:
        variants[f'clip{cl}']=np.clip(pev,anchor-cl,anchor+cl)
    for sw in [0.6,0.8]:
        variants[f'shr{sw}']=anchor+sw*(pev-anchor)
    variants['clip45_shr0.8']=anchor+0.8*(np.clip(pev,anchor-45,anchor+45)-anchor)
    return {k:rmse(yt,v) for k,v in variants.items()}, int(em.sum()), wid

np.random.seed(21); wells=sorted(glob.glob('data/train/*__horizontal_well.csv'))
test=list(np.random.choice(wells,80,replace=False))
# include the 3 public wells explicitly
for pw in ['data/train/000d7d20__horizontal_well.csv','data/train/00bbac68__horizontal_well.csv','data/train/00e12e8b__horizontal_well.csv']:
    if pw not in test: test.append(pw)
res=Parallel(n_jobs=-1)(delayed(eval_well)(w) for w in test)
res=[r for r in res if r]
keys=list(res[0][0].keys())
print("=== backtest (83 wells) pooled RMSE ===")
for k in keys:
    print(f"  {k:16s}: {pooled([(r[0][k],r[1]) for r in res]):.3f}")
print("\n=== 3 public wells ===")
pub={r[2]:r[0] for r in res if r[2] in ['000d7d20','00bbac68','00e12e8b']}
for wid,d in pub.items():
    print(f"  {wid}: const={d['const']:.2f} pf={d['pf']:.2f} clip45_shr0.8={d['clip45_shr0.8']:.2f}")
