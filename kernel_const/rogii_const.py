# ROGII — const baseline: hold last known TVT through the evaluation zone.
# Self-contained Kaggle kernel. Internet OFF. Writes submission.csv.
import os, numpy as np, pandas as pd
INP='/kaggle/input/rogii-wellbore-geology-prediction'
if not os.path.isdir(INP):
    for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction','data']:
        if os.path.isdir(c): INP=c; break
sub=pd.read_csv(f'{INP}/sample_submission.csv')
sub['well']=sub['id'].str.split('_').str[0]
sub['idx']=sub['id'].str.split('_').str[1].astype(int)
out={}
for wid in sorted(sub['well'].unique()):
    h=pd.read_csv(f'{INP}/test/{wid}__horizontal_well.csv')
    ti=h['TVT_input'].values.astype(float)
    anchor=ti[np.isfinite(ti)][-1] if np.isfinite(ti).any() else 0.0
    pred=np.where(np.isfinite(ti),ti,anchor)
    for idx in sub[sub['well']==wid]['idx'].values:
        out[f'{wid}_{idx}']=pred[idx] if idx<len(pred) else anchor
sub['tvt']=sub['id'].map(out)
sub['tvt']=sub['tvt'].ffill().fillna(sub['tvt'].median())
sub[['id','tvt']].to_csv('submission.csv',index=False)
print('wrote',len(sub),'rows; NaNs:',sub['tvt'].isna().sum())
