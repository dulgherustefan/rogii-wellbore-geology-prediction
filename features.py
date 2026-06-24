"""Feature engineering for TVT prediction. Computes per-row features usable
at inference time (no leakage from eval-zone TVT). Target = TVT - anchor."""
import pandas as pd, numpy as np

def roll(a, k):
    s = pd.Series(a)
    return s.rolling(k, min_periods=1).mean().values, s.rolling(k, min_periods=1).std().fillna(0).values

def well_features(h, tw, is_train=True):
    """h: horizontal_well df (has TVT if train). tw: typewell df sorted by TVT.
    Returns DataFrame of features for the EVAL rows, plus 'anchor' and (if train) 'y'."""
    ti = h['TVT_input'].values.astype(float)
    em = ~np.isfinite(ti)
    if em.sum() == 0:
        return None
    cut = int(np.argmax(em))
    anchor = ti[cut-1] if cut > 0 else float(np.median(tw['TVT'].values))
    md = h['MD'].values; X=h['X'].values; Y=h['Y'].values; Z=h['Z'].values
    gr = h['GR'].values.astype(float)
    grf = np.where(np.isfinite(gr), gr, np.nanmedian(gr))
    twt = tw['TVT'].values; twg = tw['GR'].values

    # GR calibration on known zone (obs ~ a*ref+b)
    ki = np.where(np.isfinite(ti))[0]
    a,b = 1.0,0.0
    if len(ki) > 30:
        kref = np.interp(ti[ki], twt, twg)
        A = np.vstack([kref, np.ones(len(kref))]).T
        sol,*_ = np.linalg.lstsq(A, grf[ki], rcond=None)
        if 0.2 < sol[0] < 5: a,b = sol
    # known lateral dip (TVT vs MD on last 60% of known)
    dip = 0.0; gr_known_mean = np.nanmean(grf[ki]) if len(ki) else np.nanmedian(grf)
    if len(ki) > 60:
        k0 = ki[int(len(ki)*0.4)]
        sel = ki[ki>=k0]
        if len(sel) > 20:
            dip = np.polyfit(md[sel], ti[sel], 1)[0]
    Z_cut = Z[cut-1] if cut>0 else Z[0]
    X_cut = X[cut-1] if cut>0 else X[0]; Y_cut=Y[cut-1] if cut>0 else Y[0]
    md_cut = md[cut-1] if cut>0 else md[0]

    # global derived series
    dZ = np.gradient(Z, md)
    incl = dZ  # vertical rate per MD
    grs = pd.Series(grf).rolling(11, min_periods=1, center=True).mean().values
    dgr = np.gradient(grs, md)
    gr_rm, gr_rs = roll(grf, 31)

    # typewell ref in observed-GR space; build grid for point match
    grid = np.arange(anchor-50, anchor+50, 0.5)
    refcal = a*np.interp(grid, twt, twg)+b

    ev = np.where(em)[0]
    # point GR match: nearest grid TVT to observed GR (ambiguous but a feature)
    diff = np.abs(grf[ev][:,None] - refcal[None,:])
    pm_idx = np.argmin(diff, axis=1)
    pm_tvt = grid[pm_idx]
    pm_res = diff[np.arange(len(ev)), pm_idx]
    # GR at anchor (what typewell predicts if we stayed)
    gr_anchor = a*np.interp(anchor, twt, twg)+b

    f = {}
    f['md_into'] = md[ev]-md_cut
    f['md_into_frac'] = (md[ev]-md_cut)/max(1.0,(md[-1]-md_cut))
    f['Z'] = Z[ev]; f['Z_dev'] = Z[ev]-Z_cut
    f['incl'] = incl[ev]
    f['X_dev']=X[ev]-X_cut; f['Y_dev']=Y[ev]-Y_cut
    f['lat_dist']=np.hypot(X[ev]-X_cut, Y[ev]-Y_cut)
    f['gr']=grf[ev]; f['grs']=grs[ev]; f['dgr']=dgr[ev]
    f['gr_dev_anchor']=grf[ev]-gr_anchor
    f['gr_rm']=gr_rm[ev]; f['gr_rs']=gr_rs[ev]
    f['pm_tvt_dev']=pm_tvt-anchor
    f['pm_res']=pm_res
    f['dip']=np.full(len(ev),dip)
    f['dip_proj']=dip*(md[ev]-md_cut)
    f['calib_a']=np.full(len(ev),a)
    f['gr_known_mean']=np.full(len(ev),gr_known_mean)
    f['anchor']=np.full(len(ev),anchor)
    df = pd.DataFrame(f)
    df['anchor_val']=anchor
    df['row']=ev
    if is_train and 'TVT' in h:
        df['y']=h['TVT'].values[ev]-anchor
    return df
