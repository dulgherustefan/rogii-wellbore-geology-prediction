"""Backtest harness: load wells once, run a predictor, report pooled RMSE."""
import pandas as pd, numpy as np, glob, time, sys
import geo

WELLS = sorted(glob.glob('data/train/*__horizontal_well.csv'))

def load(wlist):
    data = []
    for w in wlist:
        wid = w.split('/')[-1].split('__')[0]
        h = pd.read_csv(w)
        tw = pd.read_csv(f'data/train/{wid}__typewell.csv').dropna(subset=['TVT','GR']).sort_values('TVT')
        ti = h['TVT_input'].values.astype(float)
        em = ~np.isfinite(ti)
        if em.sum()==0 or np.isfinite(ti).sum()<50:
            continue
        data.append(dict(wid=wid, md=h['MD'].values, gr=h['GR'].values, ti=ti,
                         y=h['TVT'].values, em=em, cut=int(np.argmax(em)),
                         twt=tw['TVT'].values, twg=tw['GR'].values))
    return data

def pooled(res):
    return np.sqrt(sum(r**2*n for r,n in res)/sum(n for _,n in res))

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def run(data, fn, **kw):
    res=[]; t0=time.time()
    for d in data:
        pred = fn(d, **kw)
        res.append((rmse(d['y'][d['em']], pred[d['em']]), int(d['em'].sum())))
    return pooled(res), time.time()-t0, res

def const_pred(d):
    p = d['ti'].copy(); p[d['em']] = d['ti'][d['cut']-1]; return p

def dp_pred(d, **kw):
    return geo.align(d['md'], d['gr'], d['ti'], d['twt'], d['twg'], **kw)

if __name__ == '__main__':
    np.random.seed(7)
    n = int(sys.argv[1]) if len(sys.argv)>1 else 100
    sub = list(np.random.choice(WELLS, n, replace=False))
    data = load(sub)
    print(f"loaded {len(data)} wells")
    c,_,_ = run(data, const_pred)
    print(f"const pooled={c:.3f}")
