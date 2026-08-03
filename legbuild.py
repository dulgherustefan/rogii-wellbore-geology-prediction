"""Compute the conservative-gs PF leg for every TRAIN well.

Two uses, both opened up by the gs finding:
  A) retune the blend weight -- w_sub1=0.60 was tuned against the OLD leg, and the
     new leg is a different estimator, so the optimum has probably moved;
  B) hand the leg to the GBM as a FEATURE so it can learn when to trust it,
     instead of it only entering through the fixed 0.40 post-process weight.

Same construction as the kernel's inference path (gs pinned at LEG_GS, the last
LEG_SEEDS of the 96-run budget), so the train column matches what test wells get.
The PF only ever reads the known prefix, so there is no leakage.
"""
import numpy as np, pandas as pd, glob, sys, time
from joblib import Parallel, delayed
import pfsweep as S

LEG_GS = float(__import__('os').environ.get('LEG_GS', 45.0))
LEG_SEEDS = 32
N_PART = 350


def leg_for_well(wid):
    hw = pd.read_csv(f'data/train/{wid}__horizontal_well.csv')
    ev = np.flatnonzero(~np.isfinite(hw['TVT_input'].to_numpy(float)))
    if len(ev) == 0:
        return None
    P = dict(S.DEF); P['GS_LO'] = LEG_GS; P['GS_HI'] = LEG_GS
    e = S.leg_for_well(wid, P, LEG_SEEDS, N_PART, 5.0)
    if e is None:
        return None
    y = hw['TVT'].to_numpy(float)[ev]
    return pd.DataFrame({'id': [f'{wid}_{i}' for i in ev],
                         'likpf_leg': (e + y).astype(np.float32)})


if __name__ == '__main__':
    t0 = time.time()
    wids = sorted(p.split('/')[-1].split('__')[0]
                  for p in glob.glob('data/train/*__horizontal_well.csv'))
    if len(sys.argv) > 1:
        wids = wids[:int(sys.argv[1])]
    res = Parallel(n_jobs=6, verbose=1)(delayed(leg_for_well)(w) for w in wids)
    df = pd.concat([r for r in res if r is not None], ignore_index=True)
    df.to_pickle(__import__('os').environ.get('LEG_OUT', 'leg_train.pkl'))
    print(f'wrote {__import__('os').environ.get('LEG_OUT','leg_train.pkl')} gs={LEG_GS} {df.shape} for {len(wids)} wells '
          f'({time.time()-t0:.0f}s)', flush=True)
