import pandas as pd, numpy as np, glob, time, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import struct_model as sm, combined, pf
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def pooled(l): return np.sqrt(sum(r**2*n for r,n in l)/sum(n for _,n in l))
print("loading cloud..."); XY,EE,WID=sm.load_cloud(step=3); tree=cKDTree(XY)
np.random.seed(21); wells=sorted(glob.glob('data/train/*__horizontal_well.csv'))
test=list(np.random.choice(wells,60,replace=False))
for pw in ['000d7d20','00bbac68','00e12e8b']:
    p=f'data/train/{pw}__horizontal_well.csv'
    if p not in test: test.append(p)
def one(w):
    wid=w.split('/')[-1].split('__')[0]
    h=pd.read_csv(w); tw=pd.read_csv(f'data/train/{wid}__typewell.csv')
    ti=h['TVT_input'].values.astype(float); em=~np.isfinite(ti)
    if em.sum()==0: return None
    y=h['TVT'].values; anchor=ti[np.isfinite(ti)][-1]
    pe=pf.pf_ensemble(h,tw,n_seeds=48); pe_s=anchor+0.6*(pe-anchor)
    cb,info=combined.predict(h,tw,XY,EE,WID,tree,exclude=wid,n_seeds=48)
    return dict(wid=wid,n=int(em.sum()),
                const=rmse(y[em],np.full(em.sum(),anchor)),
                pf_shr06=rmse(y[em],pe_s[em]),
                combined=rmse(y[em],cb[em]),
                info=info)
t0=time.time()
res=[r for r in Parallel(n_jobs=-1)(delayed(one)(w) for w in test) if r]
print(f"{len(res)} wells {time.time()-t0:.0f}s")
for k in ['const','pf_shr06','combined']:
    print(f"  {k:10s}: pooled={pooled([(r[k],r['n']) for r in res]):.3f}")
print("\n3 public wells:")
for r in res:
    if r['wid'] in ['000d7d20','00bbac68','00e12e8b']:
        print(f"  {r['wid']}: const={r['const']:.2f} pf_shr06={r['pf_shr06']:.2f} combined={r['combined']:.2f}  w_pf={r['info']['w_pf']:.2f} w_st={r['info']['w_st']:.2f} st_kres={r['info']['st_kres']:.1f}")
