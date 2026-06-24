import pandas as pd, numpy as np, glob, time, itertools, geo4
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def pooled(l): return np.sqrt(sum(r**2*n for r,n in l)/sum(n for _,n in l))
np.random.seed(9); sub=np.random.choice(sorted(glob.glob('data/train/*__horizontal_well.csv')),70,replace=False)
data=[]
for w in sub:
    wid=w.split('/')[-1].split('__')[0]
    h=pd.read_csv(w); tw=pd.read_csv(f'data/train/{wid}__typewell.csv').dropna(subset=['TVT','GR']).sort_values('TVT')
    ti=h['TVT_input'].values.astype(float); em=~np.isfinite(ti)
    if em.sum()==0: continue
    data.append((h['MD'].values,h['GR'].values,ti,tw['TVT'].values,tw['GR'].values,h['TVT'].values,em,int(np.argmax(em))))
cbase=[]
for md,gr,ti,twt,twg,y,em,cut in data:
    cbase.append((rmse(y[em],np.full(em.sum(),ti[cut-1])),em.sum()))
print(f"{len(data)} wells, const pooled={pooled(cbase):.3f}",flush=True)
best=(99,None)
grid=list(itertools.product([5,8,12],[1.0,3.0,8.0],[0.005,0.02,0.06],[30,50]))
for s_lvl,curv,gamma,seg in grid:
    res=[]
    for md,gr,ti,twt,twg,y,em,cut in data:
        anchor=ti[cut-1]
        try:
            pred=geo4.align(md,gr,ti,twt,twg,s_lvl=s_lvl,curv=curv,gamma=gamma,seg=seg)
            blend=anchor+0.75*(pred[em]-anchor)
            res.append((rmse(y[em],blend),em.sum()))
        except Exception as e:
            res.append((rmse(y[em],np.full(em.sum(),anchor)),em.sum()))
    p=pooled(res)
    tag=f"s{s_lvl} c{curv} g{gamma} seg{seg}"
    if p<best[0]: best=(p,tag); print(f"  {tag:30s} pooled={p:.3f} *",flush=True)
print("BEST:",best,flush=True)
