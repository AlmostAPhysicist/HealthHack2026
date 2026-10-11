"""Model 1: time series regressor (ARX = autoregressive + exogenous inputs), fit by least squares.

From an anchor time t0 (last known glucose), predict t0 + h for h = 5 .. 240 min:

    g(t0+h) = g(t0)                                         start where glucose is now
            + momentum * shape(h)                           keep going the way it was heading
            + (g(t0) - baseline) * shape(h)                 drift back toward the person's normal level
            + sum over inputs of [x(t0+h) - x(t0)] * beta   food, insulin, activity, HR (exogenous.py)
            + time of day

x(t) = each input convolved with fixed response shapes. Linear in beta, so it is ridge least squares,
with population -> group -> person shrinkage (common.Hierarchy).

v5 adds more from the glucose history itself (all from before t0):
    acceleration    is the trend speeding up or flattening (last 15 min vs the 15 before)
    trend over 60   the slower trend of the last hour
    high / low      how far above 180 or below 70: the body pulls back harder at the extremes
    own rhythm      this person's usual glucose at that clock time (median over their earlier days) minus at t0:
                    their personal dawn rise, afternoon dip, ... rather than one shared sine wave
"""
import numpy as np
import pandas as pd

from common import HORIZON, Hierarchy
from exogenous import effect_curves, kernels_for, ramp, group_of

H = np.arange(1, HORIZON + 1) * 5
AR_SHAPES = {"momentum_20": ramp(20)(H), "momentum_60": ramp(60)(H),
             "revert_60": ramp(60)(H), "revert_240": ramp(240)(H)}
V5_SHAPES = {"accel_20": ramp(20)(H), "trend60_60": ramp(60)(H), "high_120": ramp(120)(H), "low_60": ramp(60)(H)}
V5 = list(V5_SHAPES) + ["own_rhythm"]
BIN = 6                                   # own rhythm in 30-min clock bins


def own_rhythm(series):
    """Each step: the person's median glucose in that 30-min clock bin over all earlier days (nan if none yet)."""
    g = series["glucose"]
    day = series.index.normalize()
    clock = np.asarray((series.index.hour * 60 + series.index.minute) // (5 * BIN))
    out = np.full(len(g), np.nan)
    days = day.unique()
    for d in days[1:]:
        past = np.asarray(day < d)
        prof = g[past].groupby(clock[past]).median()
        now = np.asarray(day == d)
        out[now] = prof.reindex(clock[now]).to_numpy()
    return out


class Regressor(Hierarchy):
    kind = "regressor"

    def __init__(self, inputs, group_strength=3000.0, person_strength=3000.0, extra=()):
        super().__init__(group_strength, person_strength)
        self.kernels = kernels_for(inputs)
        self.extra = list(extra)
        self.features = self.kernels + list(AR_SHAPES) + ["sin_day", "cos_day"] + self.extra
        self.n_features = len(self.features)

    def group_of(self, feature):
        if feature.startswith(("momentum", "accel", "trend60")):
            return "recent trend"
        if feature.startswith(("revert", "high", "low")):
            return "pull to baseline"
        if feature in ("sin_day", "cos_day", "own_rhythm"):
            return "daily rhythm"
        return group_of(feature)

    def design(self, series, anchors, base):
        """Rows for every (anchor, horizon). X (n_anchors, HORIZON, n_features), Y = glucose change from anchor."""
        g = series["glucose"].to_numpy(float)
        n = len(g)
        curves = effect_curves(series, self.kernels)
        idx = anchors[:, None] + np.arange(1, HORIZON + 1)[None, :]
        inside = idx < n
        idx = np.minimum(idx, n - 1)

        g0 = g[anchors]
        back = lambda k: g[np.maximum(anchors - k, 0)]
        momentum = np.nan_to_num(g0 - back(6))                           # change over the last 30 min
        revert = g0 - base                                             # distance from their normal level
        ang = 2 * np.pi * (series.index.hour + series.index.minute / 60).to_numpy() / 24
        sin, cos = np.sin(ang), np.cos(ang)

        parts = [curves[idx] - curves[anchors][:, None, :],
                 np.stack([momentum[:, None] * AR_SHAPES["momentum_20"], momentum[:, None] * AR_SHAPES["momentum_60"],
                           revert[:, None] * AR_SHAPES["revert_60"], revert[:, None] * AR_SHAPES["revert_240"],
                           sin[idx] - sin[anchors][:, None], cos[idx] - cos[anchors][:, None]], axis=2)]
        if getattr(self, "extra", None):
            values = {"accel_20": np.nan_to_num((g0 - back(3)) - (back(3) - back(6))),
                      "trend60_60": np.nan_to_num(g0 - back(12)),
                      "high_120": np.nan_to_num(np.maximum(g0 - 180, 0)),
                      "low_60": np.nan_to_num(np.maximum(70 - g0, 0))}
            cols = []
            for f in getattr(self, "extra", []):
                if f == "own_rhythm":
                    r = own_rhythm(series) if "own_rhythm" not in series else series["own_rhythm"].to_numpy(float)
                    r = pd.Series(r).ffill().to_numpy()
                    cols.append(np.nan_to_num(r[idx] - r[anchors][:, None]))
                else:
                    cols.append(values[f][:, None] * V5_SHAPES[f])
            parts.append(np.stack(cols, axis=2))
        X = np.concatenate(parts, axis=2)
        return X, np.where(inside, g[idx] - g0[:, None], np.nan)
