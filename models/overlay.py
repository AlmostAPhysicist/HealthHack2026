"""Model 2: overlay model. Glucose = baseline + spikes + cooldowns stacked on every meal (and insulin dose).

    g(t) = baseline part                                    level it drifts back to, recent trend, daily rhythm,
                                                            exercise and heart rate
         + sum over events k (meals, insulin doses) of
               sum_i a_i(event k) * spike_i(warped t)       3 rising sigmoids:   sigmoid((t - center_i) / width_i)
             + sum_j c_j(event k) * cooldown_j(warped t)    3 falling sigmoids: -sigmoid((t - center_j) / width_j)

Event types, each with its own 6 shapes: food (meals), insulin (type 1 doses), exercise (after a meal vs on
its own: separate sizes), quiet (4-hour blocks with no meal or exercise, one size per time of day: night,
dawn, ...; see events.py).

Each sigmoid starts at 0 at the moment of the event. center = when it happens, width = how fast.

Every meal gets its own spikes and cooldowns:
  - HOW BIG (a_i, c_j): a linear mix of the meal and its context: carbs, protein, fat, fiber, calories,
    meal number of the day, time of day, carbs/calories already eaten today, time since the last meal,
    exercise before, glucose before, the person's usual level and swings, their own past spikes
    (context.py). Learned by least squares, population -> group -> person.
  - HOW FAST (time-warp): t is divided by a per-meal speed factor exp(w . z), z = meal size, fat/protein/
    fiber share, time of day, time since last meal, exercise (context.warp_features). A big fatty dinner can
    rise later and last longer than a light breakfast.

Fitting:
    centers, widths, warp weights (non-linear) -> optimized once per backend (scipy Powell), one event type at a time
    sizes (linear)                             -> least squares with 3-layer shrinkage (common.Hierarchy)
    noise                                      -> spread of the leftover error after a meal (error band)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

import pandas as pd

from common import HORIZON, Hierarchy, Stats, solve, baseline, meal_anchors, row_variance
from context import WARP_FEATURES, context_inputs, meal_table, warp_features
from events import event_inputs
from exogenous import effect_curves

LOOKBACK = 96                      # events up to 8 h before the anchor still matter
START = {                          # starting guesses (minutes): 3 spike centers, 3 cooldown centers, then their widths
    "food":    [5.4, 24.1, 39.8, 46.0, 157.9, 108.2, 4.8, 4.7, 5.1, 18.8, 119.6, 56.9],   # v3 fit
    "insulin": [30, 75, 150, 120, 200, 280, 15, 25, 40, 30, 40, 60],
    "exercise": [15, 40, 90, 20, 60, 120, 8, 15, 25, 10, 20, 40],
    "quiet": [30, 90, 180, 60, 120, 200, 20, 30, 40, 30, 40, 60],
}
SHAPE_BOUNDS = [(5, 360)] * 6 + [(3, 120)] * 6
BASE_START, BASE_BOUNDS = [120.0, 30.0], [(30, 600), (5, 120)]      # baseline pull time, trend fade time
WARP_BOUND = 1.0                   # each warp weight in [-1, 1]: a meal can be up to ~e x faster/slower
BASE_CURVES = ["activity_b15", "activity_b60", "activity_r120", "hr_b15", "hr_b60"]   # exercise in the baseline
H = np.arange(0, HORIZON + 1)
SHAPE_NAMES = ["spike1", "spike2", "spike3", "cool1", "cool2", "cool3"]


def sigmoid_from_zero(t, center, width):
    """Sigmoid that is exactly 0 at t = 0 (the event) and rises to 1. 0 before the event."""
    s = 1 / (1 + np.exp(-(t - center) / width))
    s0 = 1 / (1 + np.exp(center / width))
    return np.where(t >= 0, (s - s0) / (1 - s0), 0.0)


def overlays(tau, p):
    """The 6 overlay shapes at times `tau` (min after the event). Returns (..., 6): 3 spikes, 3 cooldowns."""
    shapes = sigmoid_from_zero(tau[..., None], p[:6], p[6:12])
    return shapes * np.array([1, 1, 1, -1, -1, -1])


class Overlay(Hierarchy):
    kind = "overlay"

    def __init__(self, events, group_strength=3000.0, person_strength=3000.0, warp=True):
        super().__init__(group_strength, person_strength)
        self.events, self.warp = events, warp
        self.params = {e: np.array(START[e], float) for e in events}
        self.warp_w = np.zeros(len(WARP_FEATURES))
        self.base_params = np.array(BASE_START)
        self.features = [f"{e}:{c}:{s}" for e, cols in events.items() for s in SHAPE_NAMES for c in cols]
        self.features += ["pull_to_baseline", "recent_trend", "sin_day", "cos_day", *BASE_CURVES]
        self.n_features = len(self.features)

    # ----- rows, split into a slow part (once per person) and a fast part (once per shape guess) -----

    def prepare(self, series, anchors, base):
        """Everything that does not depend on the shapes: event pairs, their sizes and warp descriptors,
        baseline terms, targets."""
        g = series["glucose"].to_numpy(float)
        n = len(g)
        table = meal_table(series)
        warp_steps, Z = warp_features(series, table)
        z_at = dict(zip(warp_steps, Z))
        events = {}
        for e, (steps, U) in self.event_inputs(series, table).items():
            rel = steps[None, :] - anchors[:, None]                       # event step relative to anchor
            ai, ek = np.nonzero((rel >= -LOOKBACK) & (rel <= HORIZON))
            zero = np.zeros(len(WARP_FEATURES))
            events[e] = {"ai": ai, "rel": rel[ai, ek], "U": U[ek],
                         # which anchor each (anchor, event) pair belongs to, for a fast sparse sum
                         "A": csr_matrix((np.ones(len(ai)), (ai, np.arange(len(ai)))), shape=(len(anchors), len(ai))),
                         "Z": np.array([z_at.get(st, zero) for st in steps[ek]], float).reshape(len(ek), len(zero))}
        idx = np.minimum(anchors[:, None] + H[None, 1:], n - 1)
        g0 = g[anchors]
        ang = 2 * np.pi * (series.index.hour + series.index.minute / 60).to_numpy() / 24
        curves = effect_curves(series, BASE_CURVES)
        return {
            "n": len(anchors), "events": events,
            "revert": (base - g0)[:, None], "momentum": np.nan_to_num(g0 - g[np.maximum(anchors - 6, 0)])[:, None],
            "rhythm": np.stack([np.sin(ang)[idx] - np.sin(ang)[anchors][:, None],
                                np.cos(ang)[idx] - np.cos(ang)[anchors][:, None]], axis=2),
            "exercise": curves[idx] - curves[anchors][:, None, :],
            "Y": np.where(anchors[:, None] + H[None, 1:] < n, g[idx] - g0[:, None], np.nan),
        }

    def event_inputs(self, series, table=None):
        """{event type: (steps, inputs)}: every event and the values that set its overlays' sizes."""
        table = meal_table(series) if table is None else table
        ctx = {**context_inputs(series, table), **event_inputs(series)}
        s = series.fillna(0)
        ctx["meal"] = (s["carbs"].to_numpy() > 0).astype(float)
        out = {}
        for e, cols in self.events.items():
            V = np.column_stack([s[c].to_numpy(float) if c in s.columns else ctx[c] for c in cols])
            steps = np.flatnonzero((V != 0).any(axis=1))          # an event = any of its inputs is non-zero
            out[e] = (steps, V[steps])
        return out

    def event_design(self, series, event, step, minutes, table=None, inputs=None, warps=None):
        """One event's overlays as linear functions of the coefficients: {name: V}, curve = V @ coef,
        V (len(minutes), n_features). Names: spike 1-3, cooldown 1-3, all spikes, all cooldowns, total.
        Each sigmoid is scaled by this event's inputs, delayed, time-warped; 0 before the event."""
        table = meal_table(series) if table is None else table
        inputs = self.event_inputs(series, table) if inputs is None else inputs
        steps, U = inputs[event]
        u = U[list(steps).index(step)]
        tau = np.asarray(minutes, float)
        if event == "food" and self.warp:
            w_steps, Z = warp_features(series, table) if warps is None else warps
            if step in w_steps:
                tau = tau / np.exp(Z[list(w_steps).index(step)] @ self.warp_w)
        shapes = overlays(tau, self.params[event])                        # (T, 6)
        G = np.zeros((6, self.n_features))                                # amplitude of shape s = G[s] @ coef
        for si, sname in enumerate(SHAPE_NAMES):
            for ci, c in enumerate(self.events[event]):
                G[si, self.features.index(f"{event}:{c}:{sname}")] = u[ci]
        combos = {f"spike {i}": np.eye(6)[i - 1] for i in (1, 2, 3)}
        combos.update({f"cooldown {i}": np.eye(6)[i + 2] for i in (1, 2, 3)})
        combos.update({"all spikes": np.r_[1, 1, 1, 0, 0, 0.0], "all cooldowns": np.r_[0, 0, 0, 1, 1, 1.0],
                       "total": np.ones(6)})
        return {name: (shapes * w) @ G for name, w in combos.items()}

    def event_curves(self, series, post, event, step, minutes=np.arange(0, 361, 5)):
        """One event's overlays as they play out for this person, with sd propagated from the coefficient
        covariance. Returns (mean, sd) DataFrames indexed by minutes after the event."""
        coef, cov, _ = post
        k = self.kappa[np.clip(minutes // 5 - 1, 0, HORIZON - 1)]
        V = self.event_design(series, event, step, minutes)
        mean = {n: v @ coef for n, v in V.items()}
        sd = {n: k * np.sqrt(row_variance(v, cov)) for n, v in V.items()}
        return pd.DataFrame(mean, index=minutes), pd.DataFrame(sd, index=minutes)

    def part(self, prep, e):
        """Columns of X for one event type (or 'base'), (n_anchors, HORIZON, k)."""
        if e == "base":
            mins = H[1:] * 5.0
            pull = prep["revert"] * (1 - np.exp(-mins / self.base_params[0]))
            trend = prep["momentum"] * (1 - np.exp(-mins / self.base_params[1]))
            return np.concatenate([np.stack([pull, trend], axis=2), prep["rhythm"], prep["exercise"]], axis=2)
        ev, p = prep["events"][e], len(self.events[e])
        if not len(ev["ai"]):
            return np.zeros((prep["n"], HORIZON, 6 * p))
        tau = (H[None, :] - ev["rel"][:, None]) * 5.0                     # minutes since the event
        if e == "food" and self.warp:
            tau = tau / np.exp(ev["Z"] @ self.warp_w)[:, None]             # this meal's own speed
        S = overlays(tau, self.params[e])
        dS = (S[:, 1:] - S[:, :1]).reshape(len(tau), -1)                  # change since the anchor, (pairs, HORIZON*6)
        # sum each anchor's events, weighted by each event input: (anchors, HORIZON*6, inputs)
        Xe = np.stack([ev["A"] @ (dS * ev["U"][:, c][:, None]) for c in range(p)], axis=-1)
        return Xe.reshape(prep["n"], HORIZON, 6 * p)

    def design_from(self, prep):
        """X (n_anchors, HORIZON, n_features) for the current shapes."""
        return np.concatenate([self.part(prep, e) for e in [*self.events, "base"]], axis=2)

    def design(self, series, anchors, base):
        prep = self.prepare(series, anchors, base)
        return self.design_from(prep), prep["Y"]

    # ----- non-linear part -----

    def fit_shapes(self, people, evals_per_stage=150, log=print, food_evals=None):
        """Optimize centers, widths (log scale) and the food time-warp so pooled least squares is smallest.
        In stages: food, insulin, exercise, quiet, baseline. Each stage moves only its own shapes; the other
        parts stay cached. Fitted at every 15 min (incl. 30/60/90/120) for speed."""
        keep = np.arange(2, HORIZON, 3)                                    # 15, 30, ..., 240 min
        preps = [self.prepare(s, self.shape_anchors(s), baseline(s)) for s, _ in people if len(meal_anchors(s))]
        Ys = [p["Y"][:, keep] for p in preps]
        blocks = [*self.events, "base"]
        cache = [{e: self.part(p, e)[:, keep] for e in blocks} for p in preps]

        def loss_for(stage):
            def loss(x):
                self._set(stage, x)
                st = Stats(self.n_features)
                for p, c, y in zip(preps, cache, Ys):
                    c[stage] = self.part(p, stage)[:, keep]
                    st.add(np.concatenate([c[e] for e in blocks], axis=2), y)
                scale = np.diag(st.G) / st.n + 1e-9
                beta = solve(st, np.zeros(self.n_features), 100.0, scale)
                return float(-(2 * beta @ st.b - beta @ st.G @ beta) / st.n)   # = mean squared error minus a constant
            return loss

        for stage in blocks:
            x0, bounds = self._get(stage)
            best = minimize(loss_for(stage), x0, method="Powell", bounds=bounds,
                            options={"maxfev": food_evals if stage == "food" and food_evals else evals_per_stage,
                                     "xtol": 1e-2, "ftol": 1e-4})
            loss_for(stage)(best.x)                                       # leave the best shapes in place + cache
            log(f"  overlay shapes, stage '{stage}': loss {best.fun:.1f}")
        return self

    def _get(self, stage):
        """Current parameters of one stage (log scale) and their bounds."""
        if stage == "base":
            return np.log(self.base_params), [(np.log(lo), np.log(hi)) for lo, hi in BASE_BOUNDS]
        x, b = np.log(self.params[stage]), [(np.log(lo), np.log(hi)) for lo, hi in SHAPE_BOUNDS]
        if stage == "food" and self.warp:
            x, b = np.r_[x, self.warp_w], b + [(-WARP_BOUND, WARP_BOUND)] * len(WARP_FEATURES)
        return x, b

    def _set(self, stage, x):
        if stage == "base":
            self.base_params = np.exp(x)
        else:
            self.params[stage] = np.exp(x[:12])
            if stage == "food" and self.warp:
                self.warp_w = np.asarray(x[12:])

    def copy_shapes(self, other):
        self.params, self.base_params = dict(other.params), other.base_params
        self.warp_w = other.warp_w if self.warp else self.warp_w
        return self

    # ----- explaining -----

    def group_of(self, feature):
        """'food:carbs:spike2' -> 'spike 2', 'insulin:bolus:cool1' -> 'insulin cooldown 1', the rest -> 'baseline'."""
        if ":" not in feature:
            return "baseline"
        event, _, shape = feature.split(":")
        name = shape.replace("cool", "cooldown ").replace("spike", "spike ")
        return name if event == "food" else f"{event} {name}"   # e.g. 'exercise spike 1', 'quiet cooldown 2'

    def shape_anchors(self, series):
        """Anchors for fitting shapes: every meal + every 3 h (so exercise and quiet time are covered too)."""
        g = series["glucose"].to_numpy()
        every3h = np.arange(0, len(g), 36)
        return np.union1d(meal_anchors(series), every3h[~np.isnan(g[every3h])])

    def describe(self):
        """Fitted shapes (minutes) and warp weights, for the write-up."""
        out = {e: {"spike_centers": p[0:3], "cooldown_centers": p[3:6], "spike_widths": p[6:9], "cooldown_widths": p[9:12]}
               for e, p in self.params.items()}
        out["baseline"] = {"pull_time": self.base_params[0], "trend_fade": self.base_params[1]}
        if self.warp:
            out["meal_slowdown"] = dict(zip(WARP_FEATURES, np.exp(self.warp_w)))   # >1 = slower when that is higher
        return out
