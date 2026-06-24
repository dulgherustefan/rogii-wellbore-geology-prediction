import sys, glob, numpy as np, pandas as pd, time
sys.path.insert(0,'kernel_stack')
import rogii_stacker as K
K.PF_SEEDS=12  # fast
# limit train wells for smoke
import importlib
t0=time.time()
K.ensure_cloud(); XY=K.CL["XY"]
print(f"cloud {len(XY)} {time.time()-t0:.0f}s")
train_wids=[w.split('/')[-1].split('__')[0] for w in sorted(glob.glob(f'{K.TRAIN}/*__horizontal_well.csv'))][:40]
from joblib import Parallel,delayed
tr=[x for x in Parallel(n_jobs=-1)(delayed(K.well_feats)(w,'train') for w in train_wids) if x is not None]
df=pd.concat(tr,ignore_index=True)
print("train feats",df.shape)
import lightgbm as lgb
m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=63,min_child_samples=100,verbose=-1).fit(df[K.FEATS].values,df['y'].values)
# predict 3 test
sub=pd.read_csv(f'{K.INP}/sample_submission.csv'); sub['well']=sub['id'].str.split('_').str[0]; sub['idx']=sub['id'].str.split('_').str[1].astype(int)
te=[K.well_feats(w,'test') for w in sorted(sub['well'].unique())]
err=[]
for fte in te:
    wid=fte['wid'].iloc[0]; anchor=fte['anchor'].iloc[0]
    pr=np.clip(m.predict(fte[K.FEATS].values),-90,90)
    tvt=anchor+pr
    tr_h=pd.read_csv(f'data/train/{wid}__horizontal_well.csv')
    yt=tr_h['TVT'].values[fte['row'].values]
    e=tvt-yt; err.append(e)
    print(f"{wid}: stacker RMSE={np.sqrt(np.mean(e**2)):.2f} const={np.sqrt(np.mean((yt-anchor)**2)):.2f}")
a=np.concatenate(err); print(f"POOLED 3 wells: {np.sqrt(np.mean(a**2)):.2f} ({time.time()-t0:.0f}s total)")
print("SMOKE OK")
