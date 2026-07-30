"""Visible-prefix calibration: pick the predictor per well using its own labels.

Every well ships a known TVT prefix. Hiding the tail of that prefix gives a free,
per-well, leak-clean validation set: re-run each candidate on the shortened
prefix and score it on the part we hid. The candidate that reconstructs a well's
own recent history best is the one trusted for the eval zone.

This is the lever the public frontier notebooks use to get from ~7.2 to ~6.5,
and it needs no training data at all -- only the labels the organiser gave us.
"""
import numpy as np, pandas as pd


def _truncate(hw, frac):
    """Return a copy of the well whose known prefix is cut back to `frac`, plus
    the indices that were hidden (our scoring set)."""
    t = hw['TVT_input'].to_numpy(float)
    kn = np.flatnonzero(np.isfinite(t))
    cut = int(len(kn) * frac)
    if cut < 30 or len(kn) - cut < 20:
        return None, None
    hidden = kn[cut:]
    hw2 = hw.copy()
    t2 = t.copy(); t2[hidden] = np.nan
    hw2['TVT_input'] = t2
    return hw2, hidden


def calibrate(hw, tw, cands, cut_fracs=(0.5, 0.65, 0.8), mode='select',
              temp=0.25, min_gain=0.0):
    """Score each candidate on hidden prefix suffixes, then combine.

    mode='select' takes the single best candidate; mode='weight' takes a
    softmax-weighted average, which is steadier when the ranking is marginal.
    Returns (prediction, report).
    """
    names = list(cands)
    errs = {n: [] for n in names}
    for f in cut_fracs:
        hw2, hidden = _truncate(hw, f)
        if hw2 is None:
            continue
        truth = hw['TVT_input'].to_numpy(float)[hidden]
        for n in names:
            try:
                p = cands[n](hw2, tw)
                e = float(np.sqrt(np.nanmean((p[hidden] - truth) ** 2)))
            except Exception:
                e = np.inf
            errs[n].append(e)

    score = {n: (float(np.mean(v)) if v else np.inf) for n, v in errs.items()}
    finite = {n: s for n, s in score.items() if np.isfinite(s)}
    if not finite:
        return cands[names[0]](hw, tw), {'pick': names[0], 'scores': score}

    full = {n: cands[n](hw, tw) for n in names}
    best = min(finite, key=finite.get)
    base = names[0]                       # first candidate is the incumbent
    if mode == 'select':
        if finite.get(base, np.inf) - finite[best] < min_gain:
            best = base
        return full[best], {'pick': best, 'scores': score}

    s = np.array([finite.get(n, np.inf) for n in names], float)
    w = np.exp(-(s - np.nanmin(s)) / (temp * np.nanmin(s) + 1e-9))
    w[~np.isfinite(w)] = 0.0
    w /= max(w.sum(), 1e-12)
    pred = np.zeros_like(full[names[0]], dtype=float)
    for wt, n in zip(w, names):
        pred += wt * np.asarray(full[n], float)
    return pred, {'pick': 'weighted', 'w': dict(zip(names, w.round(3))), 'scores': score}
