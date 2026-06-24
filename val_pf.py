import pandas as pd, numpy as np, glob, time, warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
import pf
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def pooled(l): return np.sqrt(sum(r**2*n for r,n in l)/sum(n for _,n in l))
np.random.seed(21); wells=sorted(glob.glob('data/train/*__horizontal_well.csv'))
test=list(np.random.choice(wells,80,replace=False))
def one(w):
    wid=w.split('/')[-1].split('__')[0]
    h=pd.read_csv(w); tw=pd.read_csv(f'data/train/{wid}__typewell.csv')
    ti=h['TVT_input'].values.astype(float); em=~np.isfinite(ti)
    if em.sum()==0: return None
    y=h['TVT'].values; anchor=ti[np.isfinite(ti)][-1]
    pe=pf.pf_ensemble(h,tw,n_seeds=64)
    # far-shrink: blend toward anchor as we go deeper (robustness)
    return (rmse(y[em],np.full(em.sum(),anchor)),rmse(y[em],pe[em]),int(em.sum()))
t0=time.time()
out=Parallel(n_jobs=-1)(delayed(one)(w) for w in test)
out=[o for o in out if o]
rc=[(o[0],o[2]) for o in out]; rE=[(o[1],o[2]) for o in out]
print(f"{len(out)} wells {time.time()-t0:.0f}s")
print(f"const    pooled={pooled(rc):.3f}")
print(f"PF ens64 pooled={pooled(rE):.3f}")
import numpy as np
imp=np.array([o[0]-o[1] for o in out])
print(f"PF better on {(imp>0.3).sum()}/{len(out)} wells; worse on {(imp<-0.3).sum()}")
