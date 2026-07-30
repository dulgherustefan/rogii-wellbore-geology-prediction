"""Recover the per-well TVT line by matching horizontal GR to the typewell.

The oracle ladder says essentially all the recoverable error is a per-well line:
holding the last known TVT scores ~16 ft, the best constant ~8.8, the best line
~6.5. The prefix trend carries no information about the eval-zone slope, but the
gamma ray does: matching the calibrated horizontal GR against the typewell GR
profile puts a sharp minimum at the true stratigraphic position (cost 0.55 at
truth vs 0.95 ten feet away).

So we grid-search offset and slope directly against that matching cost, instead
of sampling the same posterior with a particle filter.
"""
import numpy as np
from dpalign import _affine_cal


def _prep(hw, tw, bin_rows=8):
    t_in = hw['TVT_input'].to_numpy(float)
    md = hw['MD'].to_numpy(float)
    gr = hw['GR'].to_numpy(float)
    kn = np.flatnonzero(np.isfinite(t_in) & np.isfinite(gr))
    ev = np.flatnonzero(~np.isfinite(t_in))
    if len(kn) < 30 or len(ev) < 10:
        return None
    tws = tw.sort_values('TVT')
    tt = tws['TVT'].to_numpy(float); tg = tws['GR'].to_numpy(float)
    m = np.isfinite(tt) & np.isfinite(tg)
    tt, tg = tt[m], tg[m]
    if len(tt) < 10:
        return None
    a, b = _affine_cal(gr[kn], np.interp(t_in[kn], tt, tg))
    gr_cal = a * gr + b

    ev = ev[np.argsort(md[ev])]
    nb = max(4, len(ev) // bin_rows)
    bins = np.array_split(ev, nb)
    bmd = np.array([md[ix].mean() for ix in bins])
    bgr = np.array([np.nanmedian(gr_cal[ix]) for ix in bins])
    ok = np.isfinite(bgr)
    return dict(t_in=t_in, md=md, ev=ev, tt=tt, tg=tg,
                bmd=bmd[ok], bgr=bgr[ok],
                m0=float(md[kn[-1]]), t0=float(t_in[kn[-1]]),
                kn=kn, gr_cal=gr_cal)


def line_search(hw, tw, off_lim=25.0, off_step=0.5,
                slope_lim=0.012, slope_step=0.0004,
                off_prior=12.0, slope_prior=0.006,
                bin_rows=8, topk=0, return_params=False):
    """Grid-search TVT = t0 + off + slope*(MD-m0) against the GR match cost."""
    P = _prep(hw, tw, bin_rows)
    if P is None:
        out = hw['TVT_input'].to_numpy(float).copy()
        k = np.flatnonzero(np.isfinite(out))
        out[~np.isfinite(out)] = out[k[-1]] if len(k) else 0.0
        return (out, 0.0, 0.0) if return_params else out
    bmd, bgr, tt, tg = P['bmd'], P['bgr'], P['tt'], P['tg']
    t0, m0 = P['t0'], P['m0']
    dx = bmd - m0

    offs = np.arange(-off_lim, off_lim + off_step, off_step)
    slopes = np.arange(-slope_lim, slope_lim + slope_step, slope_step)
    # candidate TVT curves: (n_off, n_slope, n_bins)
    tvt = t0 + offs[:, None, None] + slopes[None, :, None] * dx[None, None, :]
    pred_gr = np.interp(tvt, tt, tg)
    cost = np.mean(np.abs(pred_gr - bgr[None, None, :]), axis=2)
    # weak priors: do not wander far from the anchor, and keep the dip plausible
    cost = cost * (1.0 + 0.5 * (offs[:, None] / off_prior) ** 2
                        + 0.5 * (slopes[None, :] / slope_prior) ** 2)

    if topk and topk > 1:
        flat = cost.ravel()
        idx = np.argpartition(flat, topk)[:topk]
        w = np.exp(-(flat[idx] - flat[idx].min()) / (0.05 * flat[idx].min() + 1e-9))
        w /= w.sum()
        oi, si = np.unravel_index(idx, cost.shape)
        off = float(np.sum(w * offs[oi])); slope = float(np.sum(w * slopes[si]))
    else:
        oi, si = np.unravel_index(np.argmin(cost), cost.shape)
        off = float(offs[oi]); slope = float(slopes[si])

    out = P['t_in'].copy()
    ev = P['ev']
    out[ev] = t0 + off + slope * (P['md'][ev] - m0)
    return (out, off, slope) if return_params else out
