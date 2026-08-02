"""Build the cross-difference images that feed the mixture-density inverter.

Following Alyaev & Elsheikh (2022), the network sees the pairwise difference
between the typewell log sampled at candidate stratigraphic levels and the
horizontal log along the well:

    r[k, j] = typewell_GR(t0 + off_k) - GR_cal(bin_j)

Every dark band in that image is a candidate correlation. My own measurements
showed why this has to be handed to a learned model rather than optimised
directly: a free search over that surface finds trajectories with a LOWER GR
cost than the truth (8.14 vs 8.44) while scoring RMSE 12.5. The surface is
genuinely multi-modal, so the model must learn which mode is geologically
plausible instead of taking the argmin.

Targets are the true TVT offset from the anchor, sampled on the same MD bins.
"""
import numpy as np, pandas as pd, glob, os
from joblib import Parallel, delayed
from dpalign import _affine_cal

DATA = os.environ.get('ROGII_DATA', 'data')
K = 96                      # stratigraphic levels
L = 96                      # MD bins along the eval zone
HALF = 38.0                 # +/- ft around the anchor
OFFS = np.linspace(-HALF, HALF, K)


def well_ids(split='train'):
    return sorted(p.split('/')[-1].replace('__horizontal_well.csv', '')
                  for p in glob.glob(f'{DATA}/{split}/*__horizontal_well.csv'))


def build_one(wid, split='train', cut=None):
    """cut=None uses the organiser's own mask; a float in (0,1) shortens the
    known prefix, which is how we augment 773 wells into many samples."""
    hw = pd.read_csv(f'{DATA}/{split}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{DATA}/{split}/{wid}__typewell.csv')
    t = hw['TVT_input'].to_numpy(float)
    md = hw['MD'].to_numpy(float); z = hw['Z'].to_numpy(float)
    gr = hw['GR'].to_numpy(float)
    kn = np.flatnonzero(np.isfinite(t) & np.isfinite(gr))
    if len(kn) < 80:
        return None
    if cut is not None:
        n_keep = max(60, int(len(kn) * cut))
        kn = kn[:n_keep]
        ev = np.arange(kn[-1] + 1, len(md))
    else:
        ev = np.flatnonzero(~np.isfinite(t))
    if len(ev) < 60:
        return None

    tws = tw.sort_values('TVT')
    tt = tws['TVT'].to_numpy(float); tg = tws['GR'].to_numpy(float)
    m = np.isfinite(tt) & np.isfinite(tg)
    tt, tg = tt[m], tg[m]
    if len(tt) < 20:
        return None

    a, b = _affine_cal(gr[kn], np.interp(t[kn], tt, tg))
    gr_cal = a * gr + b
    t0, m0, z0 = t[kn[-1]], md[kn[-1]], z[kn[-1]]

    # bin the eval zone onto a fixed grid so every well yields the same shape
    edges = np.linspace(md[ev][0], md[ev][-1], L + 1)
    idx = np.clip(np.searchsorted(edges, md[ev], side='right') - 1, 0, L - 1)
    def binned(v):
        s = np.bincount(idx, weights=np.nan_to_num(v), minlength=L)
        c = np.bincount(idx, weights=np.isfinite(v).astype(float), minlength=L)
        return np.where(c > 0, s / np.maximum(c, 1e-9), np.nan)
    bgr = binned(gr_cal[ev]); bz = binned(z[ev]); bmd = binned(md[ev])
    if not np.isfinite(bgr).all():
        good = np.isfinite(bgr)
        if good.sum() < L // 2:
            return None
        xi = np.arange(L)
        bgr = np.interp(xi, xi[good], bgr[good])
        bz = np.interp(xi, xi[np.isfinite(bz)], bz[np.isfinite(bz)])
        bmd = np.interp(xi, xi[np.isfinite(bmd)], bmd[np.isfinite(bmd)])

    tw_lvl = np.interp(t0 + OFFS, tt, tg, left=np.nan, right=np.nan)
    valid = np.isfinite(tw_lvl)
    tw_lvl = np.where(valid, tw_lvl, 0.0)

    gs = max(float(np.nanstd(gr_cal[kn] - np.interp(t[kn], tt, tg))), 3.0)
    img = (tw_lvl[:, None] - bgr[None, :]) / gs                   # the correlation image
    img = np.clip(img, -6, 6)
    outside = np.repeat((~valid).astype(np.float32)[:, None], L, axis=1)
    # geometry the model needs: the well's own vertical move is known exactly
    dz = np.repeat(((bz - z0) / 50.0)[None, :], K, axis=0)
    ch = np.stack([img, outside, dz]).astype(np.float32)

    # scalar context
    prefix_tail = t[kn[-min(400, len(kn)):]]
    meta = np.array([
        (md[ev][-1] - m0) / 5000.0, len(kn) / 2000.0, gs / 30.0,
        float(np.std(prefix_tail)) / 10.0,
        float(np.nanstd(gr_cal[ev])) / 30.0,
        (t0 - tt.min()) / max(tt.max() - tt.min(), 1e-9),
    ], dtype=np.float32)

    out = dict(wid=wid, ch=ch, meta=meta, t0=float(t0), m0=float(m0),
               is_real=(cut is None),
               bmd=bmd.astype(np.float32), ev=ev.astype(np.int32))
    if 'TVT' in hw.columns:
        y = hw['TVT'].to_numpy(float)
        if np.isfinite(y[ev]).sum() > len(ev) // 2:
            by = binned(y[ev])
            g2 = np.isfinite(by)
            if g2.sum() >= L // 2:
                xi = np.arange(L)
                by = np.interp(xi, xi[g2], by[g2])
                out['y'] = (by - t0).astype(np.float32)     # offset from the anchor
    return out


def build_all(split='train', cuts=(None,), n_jobs=8, wids=None):
    wids = wids or well_ids(split)
    jobs = [(w, c) for w in wids for c in cuts]
    res = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(build_one)(w, split, c) for w, c in jobs)
    return [r for r in res if r is not None and ('y' in r or split == 'test')]


if __name__ == '__main__':
    import sys, pickle
    cuts = (None, 0.55, 0.7, 0.85)
    ds = build_all(cuts=cuts)
    print(f'samples={len(ds)}  ch={ds[0]["ch"].shape}  meta={ds[0]["meta"].shape}')
    with open('mdn_train.pkl', 'wb') as f:
        pickle.dump(ds, f, protocol=4)
    print('wrote mdn_train.pkl')
