"""Out-of-sample check of the PF sweep winners.

The sweep ran on 100 wells at 48 seeds and produced an implausible pattern --
PN=0.002, 0.01 and 0.02 all beat the shipped 0.005, so the baseline would be a
local worst. That smells of outlier-driven noise, so re-measure the candidates on
a DISJOINT well sample with double the seeds, and report a trimmed statistic
alongside RMSE to show how much is carried by a few diverging wells.
"""
import numpy as np, glob, sys, time
from joblib import Parallel, delayed
import pfsweep as S

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
wids = sorted(p.split('/')[-1].split('__')[0]
              for p in glob.glob('data/train/*__horizontal_well.csv'))
rng = np.random.RandomState(1)
perm = list(rng.permutation(wids))
holdout = perm[100:100 + n]          # disjoint from the sweep's first 100

CANDS = {
    'baseline': {},
    'PN=0.02': dict(PN=0.02),
    'RATE_SD=0.003': dict(RATE_SD=0.003),
    'MOM=0.9995': dict(MOM=0.9995),
    'SPREAD=1.5': dict(SPREAD=1.5),
    'combo': dict(MOM=0.9995, RATE_SD=0.003, SPREAD=1.5, PN=0.01, VN=0.001),
}


def run(name, over):
    P = dict(S.DEF); P.update(over)
    r = [x for x in Parallel(n_jobs=6)(
        delayed(S.leg_for_well)(w, P, 96, 350, 5.0) for w in holdout) if x is not None]
    per_well = np.array([np.sqrt(np.nanmean(e ** 2)) for e in r])
    pooled = np.sqrt(np.nanmean(np.concatenate(r) ** 2))
    return pooled, np.median(per_well), np.mean(np.sort(per_well)[:int(0.9 * len(per_well))])


t0 = time.time()
print(f'holdout wells = {len(holdout)} (disjoint from sweep), 96 seeds')
print(f'{"config":16s} {"pooled":>8s} {"median":>8s} {"trim90":>8s}')
for k, v in CANDS.items():
    p, m, t = run(k, v)
    print(f'{k:16s} {p:8.3f} {m:8.3f} {t:8.3f}   ({time.time()-t0:.0f}s)', flush=True)
