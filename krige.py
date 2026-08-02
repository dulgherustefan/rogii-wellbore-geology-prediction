"""artgor's closed-form kriged-ANCC predictor, measured leave-one-well-out.

Structurally the same physics as dipfield/surf2 but a different estimator:
krige the ABSOLUTE formation depth (ANCC) by IDW over per-well reference points,
then take the well's own datum from its known prefix.

    ANCC_k  = IDW(k=10, power=2) over (X, Y, ANCC) of OTHER wells
    C_well  = median(TVT_input - ANCC_k + Z)   over the known prefix
    TVT_hat = ANCC_k - Z + C_well

ANCC is absolute, so unlike S = TVT + Z there is no per-well datum mixed into the
cloud and no intercepts to fit. My earlier attempts used local plane/quadratic
fits (ill-conditioned, because cloud points lie along well paths) and a
nearest-donor (blew up past 700 ft); IDW over many wells is neither.

The well itself is always removed from the reference set, so the kriging error
seen here is the kriging error a test well would see.
"""
import numpy as np, pandas as pd, glob, sys, time
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

REF_STRIDE = 50          # one reference point per ~50 ft of trajectory


def load_refs():
    """Per-well reference points (X, Y, ANCC) plus everything needed to score."""
    refs = {}
    for p in sorted(glob.glob('data/train/*__horizontal_well.csv')):
        wid = p.split('/')[-1].split('__')[0]
        d = pd.read_csv(p, usecols=['MD', 'X', 'Y', 'Z', 'ANCC', 'TVT', 'TVT_input'])
        a = d['ANCC'].to_numpy(float)
        x = d['X'].to_numpy(float); y = d['Y'].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(x) & np.isfinite(y)
        if m.sum() < 50:
            continue
        idx = np.flatnonzero(m)[::REF_STRIDE]
        refs[wid] = dict(ref=np.column_stack([x[idx], y[idx], a[idx]]),
                         X=x, Y=y, Z=d['Z'].to_numpy(float),
                         TVT=d['TVT'].to_numpy(float),
                         TIN=d['TVT_input'].to_numpy(float))
    return refs


def krige(qx, qy, tree, vals, k=10, power=2.0):
    k_eff = min(k, len(vals))
    dist, idx = tree.query(np.column_stack([qx, qy]), k=k_eff)
    if k_eff == 1:
        return vals[idx]
    w = 1.0 / (dist ** power + 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    return (w * vals[idx]).sum(axis=1)


def score_well(wid, refs, k, power):
    d = refs[wid]
    others = np.concatenate([v['ref'] for w, v in refs.items() if w != wid])
    tree = cKDTree(others[:, :2])
    ak = krige(d['X'], d['Y'], tree, others[:, 2], k=k, power=power)
    t = d['TIN']; kn = np.flatnonzero(np.isfinite(t)); ev = np.flatnonzero(~np.isfinite(t))
    if len(kn) < 20 or len(ev) < 20:
        return None
    c_well = float(np.median(t[kn] - ak[kn] + d['Z'][kn]))
    pred = ak[ev] - d['Z'][ev] + c_well
    y = d['TVT'][ev]
    anchor = float(t[kn[-1]])
    return (pred - y), (anchor - y)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    t0 = time.time()
    refs = load_refs()
    print(f'wells with ANCC refs = {len(refs)} ({time.time()-t0:.0f}s)', flush=True)
    rng = np.random.RandomState(1)
    sub = list(rng.permutation(sorted(refs))[:n])
    for k, power in ((10, 2.0), (10, 1.0), (25, 2.0), (50, 2.0), (10, 3.0)):
        r = [x for x in Parallel(n_jobs=3)(delayed(score_well)(w, refs, k, power)
                                           for w in sub) if x]
        e = np.concatenate([a for a, _ in r]); c = np.concatenate([b for _, b in r])
        per = np.array([np.sqrt(np.nanmean(a ** 2)) for a, _ in r])
        print(f'  k={k:3d} p={power}: kriged pooled={np.sqrt(np.nanmean(e**2)):7.3f} '
              f'median={np.median(per):7.3f} | anchor pooled={np.sqrt(np.nanmean(c**2)):.3f}',
              flush=True)
