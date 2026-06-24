"""Generate submission.csv from test wells using the geosteering aligner.
Works locally (3 sample wells) and unchanged inside the Kaggle notebook
(just point DATA at /kaggle/input/...)."""
import pandas as pd, numpy as np, glob, os, sys
import geo

DATA = sys.argv[1] if len(sys.argv) > 1 else 'data'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'submission.csv'

PARAMS = dict(grid_res=0.5, band=45.0, smooth_k=7, lam=0.4, s_lvl=12.0,
              max_step=2.5, gamma=0.008, gamma_relax=2500.0)

def predict_test(data_dir, params):
    sub = pd.read_csv(f'{data_dir}/sample_submission.csv')
    sub['well'] = sub['id'].str.split('_').str[0]
    sub['idx'] = sub['id'].str.split('_').str[1].astype(int)
    out = {}
    for wid, g in sub.groupby('well'):
        h = pd.read_csv(f'{data_dir}/test/{wid}__horizontal_well.csv')
        tw = pd.read_csv(f'{data_dir}/test/{wid}__typewell.csv').dropna(subset=['TVT','GR']).sort_values('TVT')
        ti = h['TVT_input'].values.astype(float)
        pred = geo.align(h['MD'].values, h['GR'].values, ti,
                         tw['TVT'].values, tw['GR'].values, **params)
        for idx in g['idx'].values:
            out[f'{wid}_{idx}'] = pred[idx]
    sub['tvt'] = sub['id'].map(out)
    return sub[['id','tvt']]

if __name__ == '__main__':
    res = predict_test(DATA, PARAMS)
    assert res['tvt'].notna().all(), f"NaNs in prediction: {res['tvt'].isna().sum()}"
    res.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(res)} rows")
    print(res.head())
    print("tvt stats:", res['tvt'].describe().round(2).to_dict())
