"""Score physics legs inside the real post-process.

The pipeline-B post-process is already a two-leg blend:

    delta = w_learned * warmup * meta_oof + (1 - w_learned) * (physics_leg - last_known)

with the physics leg currently a raw, un-routed likpf_scale_5. The public
frontier notebook uses a routed SP45 selector there instead -- per-well choice of
PF scale, an optional beam mixin, and a convex shrink toward the anchor. This
compares them on identical OOF so the difference is the leg and nothing else.

Our own dumps (train_df_B.pkl, meta_oof.pkl, beam_train.pkl), so unpickling is safe.
"""
import numpy as np, pandas as pd, time
from scipy.signal import savgol_filter
import sp45

t0 = time.time()
df = pd.read_pickle('oof_B.pkl')          # carries meta_oof + the PF legs + target
beam = pd.read_pickle('beam_train.pkl')
df = df.merge(beam, on='id', how='left')
print(f'merged {df.shape} nan_meta={df.meta_oof.isna().sum()} nan_beam={df.beam.isna().sum()} '
      f'({time.time()-t0:.0f}s)', flush=True)

last = df['last_known_tvt'].to_numpy(float)
y_tvt = last + df['target'].to_numpy(float)
mds = df['md_since'].to_numpy(float)
meta = df['meta_oof'].to_numpy(float)
bm = df['beam'].to_numpy(float)
pos = [np.asarray(ix) for ix in df.groupby('well', sort=False).indices.values()]
PF = {f'likpf_scale_{s:g}': df[f'likpf_scale_{s:g}'].to_numpy(float)
      for s in (3, 5, 8, 12)}
code = np.array([sp45.well_code(n, z) for n, z in
                 zip(df['n_eval'].to_numpy(float), df['z_span'].to_numpy(float))])


def pooled(delta, sg_win=61):
    pred = last + delta
    out = pred.copy()
    for p in pos:
        v = pred[p]; wl = min(sg_win, len(v)); wl -= (wl % 2 == 0)
        if wl >= 5:
            out[p] = savgol_filter(v, wl, 3)
    return float(np.sqrt(np.mean((out - y_tvt) ** 2)))


def leg_routed(bin_variants):
    """Per-well selector trajectory using the routing table."""
    out = np.empty(len(df))
    for b, name in bin_variants.items():
        m = code == b
        if not m.any():
            continue
        cfg = sp45.parse_variant(name)
        base = PF[f"likpf_scale_{cfg['scale']:g}"][m]
        p = (1 - cfg['beam']) * base + cfg['beam'] * bm[m]
        out[m] = (1 - cfg['hold']) * p + cfg['hold'] * last[m]
    return out


def report(tag, leg, w_grid=(0.50, 0.55, 0.60, 0.65, 0.70), tau=85.0):
    warm = 1.0 - np.exp(-np.maximum(mds, 0.0) / tau)
    best = None
    for w in w_grid:
        d = w * warm * meta + (1 - w) * (leg - last)
        r = pooled(d)
        if best is None or r < best[1]:
            best = (w, r)
    print(f'{tag:34s} best w_learned={best[0]:.2f}  pooled CV={best[1]:.4f}', flush=True)
    return best


if __name__ == '__main__':
    print(f'const = {np.sqrt(np.mean((last - y_tvt) ** 2)):.4f}')
    report('current: raw likpf_scale_5', PF['likpf_scale_5'])
    report('routed selector (public bins)', leg_routed(sp45.SELECTOR_BIN_VARIANTS))
    for h in (0.0, 0.1, 0.2, 0.3):
        report(f'scale_8 hold={h}', (1 - h) * PF['likpf_scale_8'] + h * last)
    for bw in (0.05, 0.1, 0.2):
        leg = (1 - bw) * PF['likpf_scale_8'] + bw * bm
        report(f'scale_8 beam={bw} hold=0.2', 0.8 * leg + 0.2 * last)
    report('beam alone', bm)
    print(f'done ({time.time()-t0:.0f}s)')
