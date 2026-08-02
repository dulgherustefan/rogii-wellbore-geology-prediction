"""Robust-polynomial projection in stratigraphic-level space.

The public frontier pipeline has a layer my post-process never had. It works in
U = TVT + Z, the implied stratigraphic level, instead of TVT itself:

    U_hat(s) = U_anchor + robust_poly_deg3(s),   s = (MD - MD_anchor)/span
    TVT_proj = (1-lam) * TVT + lam * (U_hat - Z),   lam = 0.75

U is the geological surface and is genuinely smooth in position; TVT is
U minus the wellbore's own vertical motion, so smoothing TVT (what my savgol
does, window 61 over ~4000 rows) barely touches the trend. Smoothing U instead
removes the tracker's accumulated random-walk wander -- which is exactly the
per-well drift that dominates my error (level 7.08 of pooled 9.19).

Our own dumps, so unpickling is safe.
"""
import numpy as np


def robfit(s, y, deg=3, iters=4):
    """IRLS polynomial fit; Cauchy weights make it insensitive to the excursions
    a GR tracker throws when it briefly locks onto the wrong marker."""
    if len(s) < deg + 2:
        return y.copy()
    c = np.polyfit(s, y, deg)
    for _ in range(iters):
        r = y - np.polyval(c, s)
        sc = np.median(np.abs(r)) * 1.4826 + 1e-6
        c = np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return np.polyval(c, s)


def project_well(tvt, z, md, anchor_tvt, anchor_z, anchor_md, deg=3, lam=0.75):
    """Project one well's eval-zone trajectory through a robust polynomial in U."""
    span = max(float(md[-1] - anchor_md), 1e-6)
    s = (md - anchor_md) / span
    anchor_u = anchor_tvt + anchor_z
    fit = robfit(s, (tvt + z) - anchor_u, deg)
    full = (anchor_u + fit) - z
    if not np.all(np.isfinite(full)):
        return tvt
    return (1.0 - lam) * tvt + lam * full
