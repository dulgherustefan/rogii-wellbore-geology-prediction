import pandas as pd, numpy as np, glob, time, warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
FORM=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def pooled(l): return np.sqrt(sum(r**2*n for r,n in l)/sum(n for _,n in l))

wells=sorted(glob.glob('data/train/*__horizontal_well.csv'))
PW={}
for w in wells:
    wid=w.split('/')[-1].split('__')[0]
    h=pd.read_csv(w,usecols=['X','Y']+FORM).iloc[::2]
    PW[wid]=(h[['X','Y']].values.astype(float),h[FORM].values.astype(float))
allwids=list(PW.keys())
# prebuild concatenated arrays + index ranges per well
segs={}; XYlist=[]; EElist=[]; pos=0
for wid in allwids:
    xy,ee=PW[wid]; segs[wid]=(pos,pos+len(xy)); XYlist.append(xy); EElist.append(ee); pos+=len(xy)
BIGXY=np.vstack(XYlist); BIGEE=np.vstack(EElist)

def predict_E_excl(qxy, wid, k=25, ls=180.0):
    a,b=segs[wid]
    mask=np.ones(len(BIGXY),bool); mask[a:b]=False
    XY=BIGXY[mask]; EE=BIGEE[mask]
    tr=cKDTree(XY)
    d,ii=tr.query(qxy,k=k)
    Ep=np.full((len(qxy),6),np.nan)
    for j in range(len(qxy)):
        idx=ii[j]; dd=d[j]; wv=np.exp(-(dd/ls)**2)+1e-9
        for fi in range(6):
            vals=EE[idx,fi]; ok=np.isfinite(vals)
            if ok.sum()>=5: Ep[j,fi]=np.average(vals[ok],weights=wv[ok])
    return Ep

def one(wid, decay=0.02):
    h=pd.read_csv(f'data/train/{wid}__horizontal_well.csv')
    ti=h['TVT_input'].values.astype(float); em=~np.isfinite(ti); kn=np.isfinite(ti)
    if em.sum()==0: return None
    y=h['TVT'].values; Z=h['Z'].values.astype(float); anchor=ti[kn][-1]
    Ep=predict_E_excl(h[['X','Y']].values.astype(float), wid)
    preds=[]
    ki=np.where(kn)[0]
    for fi in range(6):
        Ef=Ep[:,fi]; m=kn&np.isfinite(Ef)
        if m.sum()<10: continue
        mi=np.where(m)[0]; wts=np.exp(decay*(np.arange(len(mi))-len(mi)))
        bw=np.average(ti[mi]+Z[mi]-Ef[mi],weights=wts); preds.append(bw+Ef-Z)
    if not preds: return (rmse(y[em],np.full(em.sum(),anchor)),1e9,em.sum())
    out=np.nanmean(np.array(preds),axis=0); out[~np.isfinite(out)]=anchor
    out=np.clip(out,anchor-80,anchor+80)
    kres=float(np.sqrt(np.nanmean((np.nanmean(np.array(preds),axis=0)[kn]-ti[kn])**2)))
    return (rmse(y[em],np.full(em.sum(),anchor)),rmse(y[em],out[em]),em.sum())

if __name__=='__main__':
    np.random.seed(31); test=list(np.random.choice(allwids,80,replace=False))
    t0=time.time()
    res=[r for r in Parallel(n_jobs=-1)(delayed(one)(wid) for wid in test) if r]
    rc=[(r[0],r[2]) for r in res]; rs=[(r[1],r[2]) for r in res]
    print(f"{len(res)} wells {time.time()-t0:.0f}s")
    print(f"const  pooled={pooled(rc):.3f}")
    print(f"STRUCT pooled={pooled(rs):.3f}")
    imp=np.array([r[0]-r[1] for r in res])
    print(f"struct better on {(imp>0.3).sum()}/{len(res)}; worse {(imp<-0.3).sum()}")
