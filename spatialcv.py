"""Spatially-blocked CV for neighbour-based signals.

Plain leave-one-well-out flatters anything spatial: a held-out train well is
still ringed by its neighbours, whereas a hidden test well is not necessarily.
That is exactly how iter2 improved pooled CV (9.295 -> 9.140) yet lost on the
board (8.481 -> 8.751): the denser, multi-scale surface was fitting a
neighbourhood density the test set does not have.

Here a whole spatial BLOCK is held out at once -- the target well and every train
well within `radius` of it leave the cloud together -- so the surface is
estimated from the kind of thinner, further neighbourhood a genuinely unseen
well would see.
"""
import numpy as np, pandas as pd
import dipfield


def blocked_surface(hw, wid, radius=0.0, k_wells=20, k_pts=240, **kw):
    """surface_shape with every well within `radius` of the target also removed."""
    dipfield.build_cloud()
    cent = dipfield._CENT
    if radius <= 0 or cent is None or wid not in cent.index:
        return dipfield.surface_shape(hw, wid=wid, k_wells=k_wells, k_pts=k_pts, **kw)

    cx, cy = cent.loc[wid, 'x'], cent.loc[wid, 'y']
    d = np.hypot(cent['x'].values - cx, cent['y'].values - cy)
    drop = cent.index.values[d <= radius]

    # temporarily hide the whole block from the cloud
    saved_cent, saved_cloud = dipfield._CENT, dipfield._CLOUD
    keep = ~np.isin(saved_cloud[0], drop)
    try:
        dipfield._CLOUD = tuple(a[keep] for a in saved_cloud)
        dipfield._CENT = saved_cent.drop(index=list(drop), errors='ignore')
        if len(dipfield._CENT) < k_wells:
            return dipfield.surface_shape(hw, wid=None, k_wells=max(2, len(dipfield._CENT)),
                                          k_pts=k_pts, **kw)
        return dipfield.surface_shape(hw, wid=None, k_wells=k_wells, k_pts=k_pts, **kw)
    finally:
        dipfield._CLOUD, dipfield._CENT = saved_cloud, saved_cent


def neighbour_stats(radius_list=(0, 2000, 5000, 10000)):
    """How far away the k-th nearest well is, to pick sensible block radii."""
    dipfield.build_cloud()
    cent = dipfield._CENT
    xy = cent[['x', 'y']].values
    out = {}
    for k in (1, 5, 20):
        ds = []
        for i in range(len(xy)):
            d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
            d.sort()
            ds.append(d[k])
        out[k] = float(np.median(ds))
    return out
