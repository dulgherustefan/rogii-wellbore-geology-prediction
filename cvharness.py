"""Well-level CV harness that mimics the organiser's setup exactly.

Every train well already ships with the same mask as the test wells: TVT_input
is a contiguous known PREFIX (~25% of rows) and the eval zone is the masked
SUFFIX (~75%). So for each train well we can predict the suffix from the prefix
only and score against the full TVT column. Pooled per-row RMSE = the metric.

Candidate predictors take (hw, tw) with TVT masked and return a full-length TVT
array; only rows in the eval zone are scored.
"""
import numpy as np, pandas as pd, glob, os
from joblib import Parallel, delayed

DATA = os.environ.get('ROGII_DATA', 'data')


def well_ids(split='train'):
    return sorted(p.split('/')[-1].replace('__horizontal_well.csv', '')
                  for p in glob.glob(f'{DATA}/{split}/*__horizontal_well.csv'))


def load(wid, split='train'):
    hw = pd.read_csv(f'{DATA}/{split}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{DATA}/{split}/{wid}__typewell.csv')
    return hw, tw


def eval_mask(hw):
    return hw['TVT_input'].isna().values


# ── candidate predictors ──────────────────────────────────────────────────
def robust_poly(x, y, xq, deg, iters=5):
    """IRLS (Tukey-ish) polynomial fit, robust to the odd bad reading."""
    x = np.asarray(x, float); y = np.asarray(y, float); xq = np.asarray(xq, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < deg + 2:
        return np.full(len(xq), np.nan if not len(y) else float(np.median(y)))
    x0, xs = x[0], (x.max() - x.min()) or 1.0
    xk, xa = (x - x0) / xs, (xq - x0) / xs
    c = np.polyfit(xk, y, deg)
    for _ in range(iters):
        r = y - np.polyval(c, xk)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = 1.0 / (1.0 + (r / (2.5 * s)) ** 2)
        c = np.polyfit(xk, y, deg, w=w)
    return np.polyval(c, xa)


def cand_const(hw, tw):
    """Hold the last known TVT."""
    t = hw['TVT_input'].to_numpy(float)
    k = np.flatnonzero(np.isfinite(t))
    return np.full(len(t), t[k[-1]] if len(k) else 0.0)


def cand_uproj(hw, tw, deg=1, tail=None):
    """Project the stratigraphic level U = TVT + Z, then subtract Z back.

    The high-frequency wiggle of TVT is just the trajectory (Z), which is known
    exactly in the eval zone. All the real error lives in the low-frequency
    trend of U, so that is the only thing worth extrapolating.
    """
    t = hw['TVT_input'].to_numpy(float)
    md = hw['MD'].to_numpy(float); z = hw['Z'].to_numpy(float)
    k = np.flatnonzero(np.isfinite(t) & np.isfinite(md) & np.isfinite(z))
    if len(k) < 30:
        return cand_const(hw, tw)
    if tail:
        k = k[-min(int(tail), len(k)):]
    u = t[k] + z[k]
    return robust_poly(md[k], u, md, deg) - z


def cand_uflat(hw, tw):
    """Assume the stratigraphic level U is frozen at its last known value: all
    remaining TVT movement is then pure trajectory (Z), which we know exactly."""
    t = hw['TVT_input'].to_numpy(float); z = hw['Z'].to_numpy(float)
    k = np.flatnonzero(np.isfinite(t) & np.isfinite(z))
    if not len(k):
        return cand_const(hw, tw)
    u = t[k[-1]] + z[k[-1]]
    return u - z


def cand_uflat_tail(hw, tw, tail=200):
    """Same, but anchor U on a median of the last `tail` known rows (less noisy)."""
    t = hw['TVT_input'].to_numpy(float); z = hw['Z'].to_numpy(float)
    k = np.flatnonzero(np.isfinite(t) & np.isfinite(z))
    if not len(k):
        return cand_const(hw, tw)
    kk = k[-min(tail, len(k)):]
    return float(np.median(t[kk] + z[kk])) - z


# ── oracles (use the truth; upper bounds only, never submitted) ───────────
def _oracle_u_poly(hw, deg):
    y = hw['TVT'].to_numpy(float); z = hw['Z'].to_numpy(float); md = hw['MD'].to_numpy(float)
    ev = eval_mask(hw) & np.isfinite(y)
    u = y + z
    return robust_poly(md[ev], u[ev], md, deg, iters=1) - z


def cand_oracle_shift(hw, tw):
    """Best possible rigid shift of the U-flat baseline: isolates the datum term."""
    y = hw['TVT'].to_numpy(float)
    base = cand_uflat(hw, tw)
    ev = eval_mask(hw) & np.isfinite(y)
    return base + float(np.mean(y[ev] - base[ev]))


def _oracle_t_poly(hw, deg, anchor=False):
    """Best-fit polynomial of TVT itself over the eval zone. With anchor=True the
    curve is forced through the last known TVT, i.e. only the SLOPE is free."""
    y = hw['TVT'].to_numpy(float); md = hw['MD'].to_numpy(float)
    t = hw['TVT_input'].to_numpy(float)
    ev = eval_mask(hw) & np.isfinite(y)
    if not anchor:
        return robust_poly(md[ev], y[ev], md, deg, iters=1)
    k = np.flatnonzero(np.isfinite(t))
    m0, t0 = md[k[-1]], t[k[-1]]
    dx, dy = md[ev] - m0, y[ev] - t0
    b = float(np.sum(dx * dy) / max(np.sum(dx * dx), 1e-9))   # slope through the anchor
    return t0 + b * (md - m0)


def cand_pf(hw, tw, n_particles=500, seed=42):
    import pf_lib
    p, _ = pf_lib.run_particle_filter(hw, tw, n_particles=n_particles, seed=seed)
    return p


def cand_pf_ens(hw, tw, n_seeds=16, scale=5.0, n_particles=400):
    import pf_lib
    return pf_lib.run_pf_lik_ensemble(hw, tw, n_particles=n_particles, n_seeds=n_seeds, scale=scale)


def cand_pf_shape(hw, tw, n_seeds=16, deg=1):
    """Keep only the PF's low-frequency SHAPE, re-anchored on the last known TVT.

    The oracle ladder says the datum is already close; what is missing is slope.
    So take the PF trajectory, fit a low-order trend to it over the eval zone and
    re-anchor that trend at the known prefix, discarding the PF's own datum drift.
    """
    import pf_lib
    p = pf_lib.run_pf_lik_ensemble(hw, tw, n_particles=400, n_seeds=n_seeds, scale=5.0)
    md = hw['MD'].to_numpy(float); t = hw['TVT_input'].to_numpy(float)
    ev = eval_mask(hw)
    k = np.flatnonzero(np.isfinite(t))
    if not len(k) or ev.sum() < 10:
        return p
    trend = robust_poly(md[ev], p[ev], md, deg, iters=2)
    m0, t0 = md[k[-1]], t[k[-1]]
    return trend - np.interp(m0, md, trend) + t0


CANDS = {
    'const':      cand_const,
    'pf1':        lambda h, t: cand_pf(h, t),
    'pf_ens16':   lambda h, t: cand_pf_ens(h, t, 16),
    'pf_shape_d1': lambda h, t: cand_pf_shape(h, t, 16, 1),
    'uflat':      cand_uflat,
    'uproj_d1_t600': lambda h, t: cand_uproj(h, t, 1, 600),
    # oracles: how much error each degree of freedom can remove
    'ORC_tconst': lambda h, t: _oracle_t_poly(h, 0),
    'ORC_tslope': lambda h, t: _oracle_t_poly(h, 1, anchor=True),
    'ORC_tline':  lambda h, t: _oracle_t_poly(h, 1),
    'ORC_tpoly3': lambda h, t: _oracle_t_poly(h, 3),
    'ORC_uline':  lambda h, t: _oracle_u_poly(h, 1),
}


def score_well(wid, names):
    hw, tw = load(wid)
    y = hw['TVT'].to_numpy(float)
    ev = eval_mask(hw) & np.isfinite(y)
    if ev.sum() == 0:
        return None
    out = {'wid': wid, 'n': int(ev.sum())}
    for nm in names:
        try:
            fn = CANDS[nm]
            # candidates that need to know which well they are on take a 3rd arg;
            # passing it explicitly keeps this thread-safe under Parallel
            p = fn(hw, tw, wid) if getattr(fn, 'needs_wid', False) else fn(hw, tw)
            out[nm] = float(np.sum((p[ev] - y[ev]) ** 2))
        except Exception:
            out[nm] = np.nan
    return out


def run(names=None, n_wells=None, n_jobs=8):
    names = names or list(CANDS)
    wids = well_ids()
    if n_wells:
        wids = wids[:n_wells]
    res = Parallel(n_jobs=n_jobs, prefer='threads')(delayed(score_well)(w, names) for w in wids)
    df = pd.DataFrame([r for r in res if r])
    tot = df['n'].sum()
    print(f'wells={len(df)}  eval rows={tot}')
    for nm in names:
        print(f'  {nm:16s} pooled RMSE = {np.sqrt(df[nm].sum()/tot):7.3f}')
    return df


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    run(n_wells=n)
