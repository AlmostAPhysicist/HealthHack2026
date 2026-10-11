"""Clean CGM glucose: remove jitter.

A wiggle counts as jitter when glucose turns around by less than `rel` (1 %) of its level before
continuing. Real turning points (bigger than that) are kept. Between two real turning points glucose
is made monotone: a rising stretch never dips, a falling stretch never bumps up.

Models train on the cleaned glucose. Scores are always against the raw readings.
"""
import numpy as np
import pandas as pd


def turning_points(x, rel=0.01, floor=1.0):
    """Indices of real local minima/maxima: each one differs from the previous by more than
    max(rel * level, floor) mg/dL. Classic zigzag filter."""
    keep, ext, direction = [0], 0, 0               # direction: +1 rising, -1 falling, 0 not yet known
    for i in range(1, len(x)):
        thr = max(rel * x[ext], floor)
        if direction == 0:
            if abs(x[i] - x[0]) > thr:
                direction, ext = (1 if x[i] > x[0] else -1), i
        elif direction * (x[i] - x[ext]) > 0:      # still going the same way: new extreme
            ext = i
        elif abs(x[i] - x[ext]) > thr:             # turned around by more than the threshold: real turning point
            keep.append(ext)
            direction, ext = -direction, i
    keep.append(len(x) - 1)
    return np.unique(keep)


def remove_jitter(glucose, rel=0.01):
    """Cleaned copy of a glucose Series. Gaps (NaN) are left as they are; each run is cleaned separately."""
    g = glucose.to_numpy(float).copy()
    ok = ~np.isnan(g)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        x = g[start:stop]
        if len(x) < 3:
            continue
        tp = turning_points(x, rel)
        for a, b in zip(tp[:-1], tp[1:]):
            seg = x[a:b + 1]
            mono = np.maximum.accumulate(seg) if seg[-1] >= seg[0] else np.minimum.accumulate(seg)
            x[a:b + 1] = np.clip(mono, min(seg[0], seg[-1]), max(seg[0], seg[-1]))
        g[start:stop] = x
    return pd.Series(g, index=glucose.index)


def clean(series, rel=0.01):
    """Series with cleaned 'glucose' and the original kept as 'glucose_raw'."""
    s = series.copy()
    s["glucose_raw"] = s["glucose"]
    s["glucose"] = remove_jitter(s["glucose"], rel)
    return s
