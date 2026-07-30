"""Per-well features for predicting the TVT line (offset + slope).

Why per-well and not per-row: the oracle ladder says the whole recoverable
signal is one line per well. Holding the last known TVT scores ~16 ft, the best
constant ~8.8, the best line ~6.4. And matching GR to the typewell on its own is
degenerate -- a free search finds trajectories with a LOWER GR cost than the
truth. So the GR evidence is summarised into features and a model learns the
prior over which minimum is the plausible one.

Target: off = mean(TVT_true - t0) over the eval zone, slope = LS slope through
the anchor. Prediction is TVT = t0 + off + slope*(MD - m0).
"""
import numpy as np, pandas as pd, glob, os
from joblib import Parallel, delayed
from dpalign import _affine_cal

DATA = os.environ.get('ROGII_DATA', 'data')
OFFS = np.arange(-30, 30.01, 0.5)
SLOPES = np.arange(-0.012, 0.01201, 0.0008)


def _cost_surface(bmd, bgr, tt, tg, t0, m0, offs=OFFS, slopes=SLOPES):
    """Mean |GR_cal - typewell_GR(TVT)| for every (offset, slope) line."""
    dx = bmd - m0
    tvt = t0 + offs[:, None, None] + slopes[None, :, None] * dx[None, None, :]
    return np.mean(np.abs(np.interp(tvt, tt, tg) - bgr[None, None, :]), axis=2)


def well_features(wid, split='train', bin_rows=8):
    hw = pd.read_csv(f'{DATA}/{split}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{DATA}/{split}/{wid}__typewell.csv')
    t = hw['TVT_input'].to_numpy(float); md = hw['MD'].to_numpy(float)
    z = hw['Z'].to_numpy(float); gr = hw['GR'].to_numpy(float)
    x = hw['X'].to_numpy(float); y = hw['Y'].to_numpy(float)
    kn = np.flatnonzero(np.isfinite(t) & np.isfinite(gr))
    ev = np.flatnonzero(~np.isfinite(t))
    if len(kn) < 100 or len(ev) < 50:
        return None
    tws = tw.sort_values('TVT')
    tt = tws['TVT'].to_numpy(float); tg = tws['GR'].to_numpy(float)
    m = np.isfinite(tt) & np.isfinite(tg); tt, tg = tt[m], tg[m]
    if len(tt) < 20:
        return None

    a, b = _affine_cal(gr[kn], np.interp(t[kn], tt, tg))
    gr_cal = a * gr + b
    t0 = float(t[kn[-1]]); m0 = float(md[kn[-1]]); z0 = float(z[kn[-1]])

    ev = ev[np.argsort(md[ev])]
    nb = max(6, len(ev) // bin_rows)
    bins = np.array_split(ev, nb)
    bmd = np.array([md[ix].mean() for ix in bins])
    bgr = np.array([np.nanmedian(gr_cal[ix]) for ix in bins])
    ok = np.isfinite(bgr) & np.isfinite(bmd)
    bmd, bgr = bmd[ok], bgr[ok]
    if len(bmd) < 6:
        return None

    C = _cost_surface(bmd, bgr, tt, tg, t0, m0)
    f = {'wid': wid, 'n_eval': int(len(ev)), 'n_known': int(len(kn))}

    # --- global minimum of the cost surface and how sharp / unique it is ---
    oi, si = np.unravel_index(np.argmin(C), C.shape)
    f['c_off'] = float(OFFS[oi]); f['c_slope'] = float(SLOPES[si])
    f['c_min'] = float(C[oi, si]); f['c_mean'] = float(C.mean())
    f['c_contrast'] = float(C.mean() - C.min())
    # profile at zero slope: constant-shape evidence for the datum
    z_idx = int(np.argmin(np.abs(SLOPES)))
    c0 = C[:, z_idx]
    f['c0_off'] = float(OFFS[np.argmin(c0)])
    f['c0_min'] = float(c0.min()); f['c0_at0'] = float(c0[np.argmin(np.abs(OFFS))])
    f['c0_gain'] = f['c0_at0'] - f['c0_min']
    # how flat is the surface along each axis at the optimum (identifiability)
    f['c_off_spread'] = float(np.std(OFFS[np.argsort(C.min(axis=1))[:8]]))
    f['c_slope_spread'] = float(np.std(SLOPES[np.argsort(C.min(axis=0))[:8]]))
    # marginal (soft-min) estimates -- less brittle than the hard argmin
    for tau in (0.05, 0.15):
        w = np.exp(-(C - C.min()) / (tau * C.min() + 1e-9)); w /= w.sum()
        f[f'm_off_{tau}'] = float((w.sum(axis=1) * OFFS).sum())
        f[f'm_slope_{tau}'] = float((w.sum(axis=0) * SLOPES).sum())

    # --- prefix geometry (no truth involved) ---
    for N in (100, 300, 1000):
        kk = kn[-min(N, len(kn)):]
        xx = md[kk] - md[kk].mean(); yy = t[kk] - t[kk].mean()
        f[f'pre_slope{N}'] = float(xx @ yy / max(xx @ xx, 1e-9))
        f[f'pre_tstd{N}'] = float(np.std(t[kk]))
    f['t0'] = t0
    f['md_span'] = float(md[ev].max() - m0)
    f['z_drift'] = float(np.nanmean(z[ev]) - z0)
    zx = md[ev] - m0; zz = z[ev] - z0
    f['z_slope'] = float(zx @ zz / max(zx @ zx, 1e-9))
    f['xy_span'] = float(np.hypot(x[ev].max() - x[ev].min(), y[ev].max() - y[ev].min()))
    f['gr_a'] = a; f['gr_b'] = b
    f['gr_resid'] = float(np.nanstd(gr_cal[kn] - np.interp(t[kn], tt, tg)))
    f['gr_ev_std'] = float(np.nanstd(gr_cal[ev])); f['gr_kn_std'] = float(np.nanstd(gr_cal[kn]))
    f['gr_ev_mean_dev'] = float(np.nanmean(gr_cal[ev]) - np.nanmean(gr_cal[kn]))

    # --- typewell sensitivity around the anchor: can GR even tell us up/down? ---
    for d in (5, 15):
        up = np.interp(t0 + d, tt, tg); dn = np.interp(t0 - d, tt, tg)
        f[f'tw_grad{d}'] = float((up - dn) / (2 * d))
        f[f'tw_absgrad{d}'] = float(abs(up - dn) / (2 * d))
    f['tw_std'] = float(np.std(tg)); f['tw_span'] = float(tt.max() - tt.min())
    f['t0_in_tw'] = float((t0 - tt.min()) / max(tt.max() - tt.min(), 1e-9))

    # --- targets (train only) ---
    if 'TVT' in hw.columns and np.isfinite(hw['TVT'].to_numpy(float)[ev]).any():
        yy = hw['TVT'].to_numpy(float)
        m2 = np.isfinite(yy[ev])
        dxe = md[ev][m2] - m0; dye = yy[ev][m2] - t0
        f['y_off'] = float(np.mean(dye))
        f['y_slope'] = float(dxe @ dye / max(dxe @ dxe, 1e-9))
    return f


def build(split='train', n=None, n_jobs=8):
    wids = sorted(p.split('/')[-1].replace('__horizontal_well.csv', '')
                  for p in glob.glob(f'{DATA}/{split}/*__horizontal_well.csv'))
    if n:
        wids = wids[:n]
    res = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(well_features)(w, split) for w in wids)
    return pd.DataFrame([r for r in res if r])


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    df = build(n=n)
    df.to_pickle('wellfeat.pkl')
    print(df.shape, '->', 'wellfeat.pkl')
    print(df[['c_off', 'c_slope', 'y_off', 'y_slope']].describe().round(4))
