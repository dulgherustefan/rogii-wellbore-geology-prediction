import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
FORM=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]

def load_cloud(step=2, train_dir='data/train'):
    cloud={}
    for w in sorted(glob.glob(f'{train_dir}/*__horizontal_well.csv')):
        wid=w.split('/')[-1].split('__')[0]
        h=pd.read_csv(w,usecols=['X','Y']+FORM).iloc[::step]
        cloud[wid]=(h[['X','Y']].values.astype(float),h[FORM].values.astype(float))
    allXY=np.vstack([c[0] for c in cloud.values()])
    allEE=np.vstack([c[1] for c in cloud.values()])
    allWID=np.concatenate([np.array([wid]*len(c[0])) for wid,c in cloud.items()])
    return allXY,allEE,allWID,cKDTree(allXY)

def predict_E(qxy, allXY,allEE,allWID,tree, exclude=None, k=25, ls=180.0, kq=400):
    d,ii=tree.query(qxy,k=min(kq,len(allXY)))
    n=len(qxy); Ep=np.full((n,6),np.nan); cov=np.full(n,np.inf)
    for j in range(n):
        idx=ii[j]; dd=d[j]
        if exclude is not None:
            m=allWID[idx]!=exclude; idx=idx[m]; dd=dd[m]
        idx=idx[:k]; dd=dd[:k]
        if len(idx)<6: continue
        cov[j]=dd[0]
        wv=np.exp(-(dd/ls)**2)+1e-9
        for fi in range(6):
            vals=allEE[idx,fi]; ok=np.isfinite(vals)
            if ok.sum()>=5: Ep[j,fi]=np.average(vals[ok],weights=wv[ok])
    return Ep,cov

def predict_tvt(h, allXY,allEE,allWID,tree, exclude=None, **kw):
    ti=h['TVT_input'].values.astype(float); Z=h['Z'].values.astype(float); kn=np.isfinite(ti)
    anchor=ti[kn][-1] if kn.any() else 0.0
    Ep,cov=predict_E(h[['X','Y']].values.astype(float), allXY,allEE,allWID,tree, exclude, **kw)
    preds=[]
    for fi in range(6):
        Ef=Ep[:,fi]; m=kn&np.isfinite(Ef)
        if m.sum()<10: continue
        bw=np.median(ti[m]+Z[m]-Ef[m]); preds.append(bw+Ef-Z)
    if not preds: return np.full(len(h),anchor),cov,1e9
    out=np.nanmean(np.array(preds),axis=0); out[~np.isfinite(out)]=anchor
    kres=float(np.sqrt(np.nanmean((out[kn]-ti[kn])**2)))
    return out,cov,kres
