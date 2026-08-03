"""The SP45 selector trajectory -- the leg of the public frontier blend I never built.

My pipeline B is the *learned* half of a two-leg blend (final = 0.60 * selector +
0.40 * learned). Everything the selector needs was already in the kernel, but only
as GBM features: the likelihood-weighted PF at four GR-noise scales and the beam
ensemble. This module assembles them into an actual trajectory and scores it.

The PF legs come free from train_df_B.pkl (96 seeds, already computed); only the
beam path has to be built here.
"""
import numpy as np, pandas as pd, glob, os
from joblib import Parallel, delayed

DATA = os.environ.get('ROGII_DATA', 'data')

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5), (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2), (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]

SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2', 1: 'pf_scale_3_hold_0.15',
    2: 'pf_scale_12_beam_0.2_hold_0.15', 3: 'pf_scale_5_hold_0.15',
    4: 'pf_scale_5_beam_0.05_hold_0.05', 5: 'pf_scale_12_beam_0.2_hold_0.05',
}
SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'


def load_well(wid, split='train'):
    hw = pd.read_csv(f'{DATA}/{split}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{DATA}/{split}/{wid}__typewell.csv')
    return hw, tw


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n = len(hgr)
    if n == 0:
        return np.array([], dtype=np.float32)
    grid = np.arange(tw_tvt.min() - 50, tw_tvt.max() + 50, 0.5)
    tw_interp = np.interp(grid, tw_tvt, tw_gr)
    lo, hi = tw_tvt.min() - es, tw_tvt.max() + es
    step = mc / max(bs, 1)
    ks = np.arange(-r, r + 1) * step
    out = np.empty(n, dtype=np.float32)
    cur = last_tvt
    gmin, gmax, glen = grid[0], grid[-1], len(grid)
    for i in range(n):
        cand = cur + ks
        idx = np.clip(((cand - gmin) / 0.5 + 0.5).astype(np.int64), 0, glen - 1)
        cur = float(cand[np.argmin(np.abs(hgr[i] - tw_interp[idx]))])
        cur = min(max(cur, lo), hi)
        out[i] = cur
    return out


def beam_rows_for_well(wid, split='train'):
    hw, tw = load_well(wid, split)
    ev_idx = hw.index[hw['TVT_input'].isna()]
    if len(ev_idx) == 0:
        return None
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].to_numpy(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).to_numpy(float)
    kn = hw[hw['TVT_input'].notna()]
    last_tvt = float(kn['TVT_input'].iloc[-1])
    gr_full = hw['GR'].astype(float).interpolate(limit_direction='both') \
                      .fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev_idx[0]:].to_numpy(np.float32)
    paths = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, *c) for c in BEAM_CONFIGS]
    ens = np.mean(np.stack(paths, 0), 0)[:len(ev_idx)]
    z = hw['Z'].to_numpy(float)
    return pd.DataFrame({'id': [f'{wid}_{i}' for i in ev_idx],
                         'beam': ens.astype(np.float32),
                         'z_span': np.float32(z.max() - z.min()),
                         'n_eval': np.int32(len(ev_idx))})


def build_beam(wids, split='train', n_jobs=6):
    res = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(beam_rows_for_well)(w, split) for w in wids)
    return pd.concat([r for r in res if r is not None], ignore_index=True)


def well_code(n_eval, z_span):
    if n_eval < SELECTOR_N_EVAL_THRESHOLD:
        return 0 if z_span < SELECTOR_Z_SPAN_THRESHOLDS[0] else 1
    return 2 if z_span < SELECTOR_Z_SPAN_THRESHOLDS[1] else 3


def parse_variant(name):
    parts = name.split('_')
    out = {'scale': 5.0, 'beam': 0.0, 'hold': 0.0}
    for key in ('scale', 'beam', 'hold'):
        if key in parts:
            out[key] = float(parts[parts.index(key) + 1])
    return out


def apply_variant(name, pf_by_scale, beam, last_known):
    """Note the hold term is a convex shrink of the WHOLE path toward the
    anchor, not a replacement of the first rows -- that shrinkage is what makes
    the selector a low-variance estimator."""
    cfg = parse_variant(name)
    base = pf_by_scale[f"likpf_scale_{cfg['scale']:g}"]
    pred = (1.0 - cfg['beam']) * base + cfg['beam'] * beam
    return (1.0 - cfg['hold']) * pred + cfg['hold'] * last_known
