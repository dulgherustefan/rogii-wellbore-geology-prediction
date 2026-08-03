"""Posterior over a smooth low-dimensional U(s), instead of a random-walk PF.

Measured facts this is built on:
  * A cubic through the anchor, dU = a1*s + a2*s^2 + a3*s^3, reproduces the true
    stratigraphic level to RMSE 2.82 over 773 wells. The family is not the limit.
  * My pipeline sits at 9.01 inside that family, and its coefficient errors are
    unbiased noise, so no post-hoc shrinkage is left.
  * Imposing smoothness AFTER inference (proj.py) bought 0.18 CV / 0.148 LB.

So impose it as the PRIOR instead. The particle filter explores a high-dimensional
random walk whose posterior mean is dominated by drift and diffusion; here the
prior is the empirical distribution of the true coefficients over the training
wells, the likelihood is the GR-vs-typewell match along the implied trajectory,
and the estimate is the posterior MEAN. The mean matters: my earlier measurements
showed the GR cost surface is multi-modal and its argmin is worse than the truth,
but a mean weighted across modes is exactly what RMSE rewards.

The prior is fitted on all train wells; a single well contributes 1/773 to it, so
evaluating on train wells is not a meaningful leak.
"""
import numpy as np, pandas as pd, os

DATA = os.environ.get('ROGII_DATA', 'data')


def fit_prior(coef):
    """Empirical Gaussian prior over (a1, a2, a3)."""
    return coef.mean(0), np.cov(coef, rowvar=False)


def sample_prior(mean, cov, n, rng, inflate=1.0):
    L = np.linalg.cholesky(cov * inflate ** 2 + 1e-6 * np.eye(len(mean)))
    return mean + rng.standard_normal((n, len(mean))) @ L.T


def posterior_curve(hw, tw, mean, cov, n_samp=4000, stride=8, temp=1.0,
                    inflate=1.2, seed=0, return_std=False):
    """Weighted-mean trajectory over cubic U-curves scored by GR match."""
    t = hw['TVT_input'].to_numpy(float)
    md = hw['MD'].to_numpy(float); z = hw['Z'].to_numpy(float)
    kn = np.flatnonzero(np.isfinite(t)); ev = np.flatnonzero(~np.isfinite(t))
    if len(kn) < 30 or len(ev) < 20:
        return None
    tws = tw.sort_values('TVT')
    tw_tvt = tws['TVT'].to_numpy(float)
    tw_gr = tws['GR'].fillna(tws['GR'].mean()).to_numpy(float)
    gr = hw['GR'].astype(float).interpolate(limit_direction='both') \
                 .fillna(float(np.nanmean(tw_gr))).to_numpy(float)

    a = kn[-1]
    U0 = t[a] + z[a]
    span = max(md[ev][-1] - md[a], 1e-6)
    s_all = (md[ev] - md[a]) / span

    # GR noise scale from the known prefix, same estimator the PF uses
    tw_at_k = np.interp(t[kn], tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(np.nan_to_num(gr[kn]) - tw_at_k), 10., 60.))

    sub = np.arange(0, len(ev), stride)
    s = s_all[sub]
    zs = z[ev][sub]
    grs = gr[ev][sub]

    rng = np.random.default_rng(seed)
    A = sample_prior(mean, cov, n_samp, rng, inflate)          # (n, 3)
    B = np.column_stack([s, s ** 2, s ** 3])                   # (m, 3)
    U = U0 + A @ B.T                                           # (n, m)
    TV = U - zs[None, :]
    np.clip(TV, tw_tvt[0] - 100, tw_tvt[-1] + 100, out=TV)
    EG = np.interp(TV, tw_tvt, tw_gr)
    d = (grs[None, :] - EG) / gs
    nll = 0.5 * np.minimum(d ** 2, 600.).sum(1) / max(temp, 1e-6)
    nll -= nll.min()
    w = np.exp(-nll)
    sw = w.sum()
    if not np.isfinite(sw) or sw <= 0:
        return None
    w /= sw
    a_mean = w @ A
    Ball = np.column_stack([s_all, s_all ** 2, s_all ** 3])
    pred = (U0 + Ball @ a_mean) - z[ev]
    if return_std:
        a2 = w @ (A ** 2)
        return pred, ev, np.sqrt(np.maximum(a2 - a_mean ** 2, 0.0))
    return pred, ev
