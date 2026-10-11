"""Shared pieces: the common 5-min patient format, CSV upload, train/test day split, least squares helpers.

Every patient = a 5-min time series + a small info dict.
Series columns (all optional except the time index):
    glucose                                 mg/dL (CGM)
    carbs, protein, fat, fiber, calories    grams / kcal eaten in that 5-min step
    bolus, basal                            insulin units delivered in that step (insulin users only)
    activity                                activity in that step (kcal burned, or steps/100)
    hr                                      heart rate, bpm
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WEIGHTS = ROOT / "models" / "weights"

STEP = "5min"
COLUMNS = ["glucose", "carbs", "protein", "fat", "fiber", "calories", "bolus", "basal", "activity", "hr"]
SUMMED = ["carbs", "protein", "fat", "fiber", "calories", "bolus", "basal", "activity"]
HORIZON = 48                      # predict up to 48 steps = 4 h after an anchor
MEAL_STEPS = [6, 12, 18, 24]      # 30 / 60 / 90 / 120 min after a meal: the scored points


# ---------- Data format ----------

def to_5min(df):
    """Put any time-indexed frame on the 5-min grid: average glucose/HR, sum food/insulin/activity."""
    df = df.sort_index()
    out = pd.DataFrame(index=pd.date_range(df.index.min().floor(STEP), df.index.max().floor(STEP), freq=STEP))
    for col in COLUMNS:
        if col not in df:
            out[col] = np.nan if col in ("glucose", "hr") else 0.0
        elif col in SUMMED:
            out[col] = df[col].fillna(0).resample(STEP).sum().reindex(out.index, fill_value=0)
        else:
            out[col] = df[col].resample(STEP).mean().reindex(out.index)
    return out


def from_csv(file):
    """Upload -> series. Any CSV parse.py understands (ours, CGMacros, meter exports); needs some time column."""
    from clean import clean
    from parse import read_frame
    df = read_frame(file)
    return clean(to_5min(df[[c for c in COLUMNS if c in df]]))


def split_days(series, days=7):
    """Days 1..days -> history, the rest -> future. Day 1 starts at midnight of the first reading."""
    cut = series.index[0].normalize() + pd.Timedelta(days=days)
    return series[series.index < cut], series[series.index >= cut]


def baseline(series):
    """A person's normal glucose level: median of what we have seen."""
    m = series["glucose"].median()
    return 100.0 if np.isnan(m) else float(m)


# ---------- Anchors (times we predict from) ----------

def meal_anchors(series):
    """Steps where food was eaten and glucose is known."""
    g = series["glucose"].to_numpy()
    return np.flatnonzero((series["carbs"].to_numpy() > 0) & ~np.isnan(g))


def hourly_anchors(series):
    """Every hour with known glucose. Teaches what glucose does between meals."""
    a = np.arange(0, len(series), 12)
    return a[~np.isnan(series["glucose"].to_numpy()[a])]


# ---------- Least squares with shrinkage (used by both model kinds) ----------

class Stats:
    """X'WX, X'Wy and y'Wy. Adding or removing someone's data is just + or -."""

    def __init__(self, k):
        self.G, self.b, self.yy, self.n = np.zeros((k, k)), np.zeros(k), 0.0, 0.0

    def add(self, X, Y, w=1.0):
        ok = ~np.isnan(Y)
        self.G += w * X[ok].T @ X[ok]
        self.b += w * X[ok].T @ Y[ok]
        self.yy += w * Y[ok] @ Y[ok]
        self.n += w * ok.sum()
        return self

    def combine(self, o, sign):
        s = Stats(len(self.b))
        s.G, s.b, s.yy, s.n = self.G + sign * o.G, self.b + sign * o.b, self.yy + sign * o.yy, self.n + sign * o.n
        return s

    def __add__(self, o):
        return self.combine(o, 1)

    def __sub__(self, o):
        return self.combine(o, -1)


def solve(stats, prior, strength, scale):
    """min |y - Xb|^2 + strength * sum scale_j (b_j - prior_j)^2. `strength` = how many rows the prior is worth."""
    P = strength * np.diag(scale)
    return np.linalg.solve(stats.G + P, stats.b + P @ prior)


MIN_ROWS = 200                    # below this many own rows, a person's noise level falls back to the population's


def residual_variance(stats, coef):
    if not hasattr(stats, "yy"):                                      # summaries saved before y'y was kept
        return np.nan
    return max(stats.yy - 2 * coef @ stats.b + coef @ stats.G @ coef, 0.0) / max(stats.n, 1.0)


def posterior(stats, prior, strength, scale, s2_population):
    """Same fit, plus its uncertainty. Treating the shrinkage as a Gaussian prior, the coefficients have
    covariance s2 * (X'WX + P)^-1, where s2 is the person's residual variance. With little own data, s2 is
    blended toward the population's residual variance (never left at 0).
    Returns (coef, cov, s2). The group prior's own (small) uncertainty is not included."""
    P = strength * np.diag(scale)
    A = stats.G + P
    coef = np.linalg.solve(A, stats.b + P @ prior)
    w = min(stats.n / MIN_ROWS, 1.0)
    own = residual_variance(stats, coef)
    s2 = s2_population if np.isnan(own) else w * own + (1 - w) * s2_population
    return coef, s2 * np.linalg.inv(A), s2


def row_variance(X, cov):
    """Variance of X @ coef for every row: x' cov x. X (..., k)."""
    return np.einsum("...k,kl,...l->...", X, cov, X)


class Hierarchy:
    """Population -> group -> person shrinkage on top of any model that predicts glucose change from anchors.
    Subclasses set self.n_features and implement design(series, anchors, base) -> X (n, HORIZON, k), Y (n, HORIZON)."""

    hourly_weight = 0.3
    # Person-specific features (meal slot, past spikes) can be shrunk less, so "this person spikes hardest at
    # dinner" survives instead of being averaged away toward the group. The factor is tuned.
    LOOSE = ("carbs_slot", "expected_rise", "carbs_x_")
    LOOSE_FACTOR = 0.2
    # Calibration of the error bands per horizon (set by train.py on the tuning days). 1 = uncalibrated.
    kappa = np.ones(HORIZON)

    def __init__(self, group_strength=3000.0, person_strength=3000.0):
        self.group_strength, self.person_strength = group_strength, person_strength

    def group_of(self, feature):
        """Readable group name for a feature (subclasses override). Used to explain predictions."""
        return feature

    def rows(self, series, base):
        """All training rows of one person: (anchors, X, Y, weight) for meal anchors and hourly anchors."""
        out = []
        for anchors, w in [(meal_anchors(series), 1.0), (hourly_anchors(series), self.hourly_weight)]:
            if len(anchors):
                out.append((anchors, *self.design(series, anchors, base), w))
        return out

    def stats(self, series, base, until=None, rows=None):
        """Least squares stats from one person. `until` = only use glucose seen before this step (no peeking).
        Pass precomputed `rows` to skip rebuilding them (meal-by-meal test)."""
        st = Stats(self.n_features)
        for anchors, X, Y, w in (self.rows(series, base) if rows is None else rows):
            if until is not None:
                Y = np.where(anchors[:, None] + np.arange(1, HORIZON + 1)[None, :] < until, Y, np.nan)
            st.add(X, Y, w)
        return st

    def stats_until(self, rows, untils):
        """stats(until=u) for several sorted `untils` at once. Each reading is added once, to the first `until` that
        may see it, and the totals are summed forward: one pass over the data instead of one per `until`."""
        untils = np.asarray(untils)
        parts = [Stats(self.n_features) for _ in untils]
        for anchors, X, Y, w in rows:
            first = np.searchsorted(untils, anchors[:, None] + np.arange(1, HORIZON + 1)[None, :], side="right")
            for k in np.unique(first[~np.isnan(Y)]):
                if k < len(untils):
                    parts[k].add(X, np.where(first == k, Y, np.nan), w)
        out, total = [], Stats(self.n_features)
        for part in parts:
            total = total + part
            out.append(total)
        return out

    def predict_from(self, series, anchors, coef, base):
        """Glucose (mg/dL) for the next HORIZON steps after each anchor. Shape (n_anchors, HORIZON)."""
        anchors = np.atleast_1d(anchors)
        X, _ = self.design(series, anchors, base)
        return series["glucose"].to_numpy()[anchors][:, None] + X @ coef

    def predict_with_sd(self, series, anchors, post, base):
        """(prediction, sd) for the next HORIZON steps after each anchor. sd = calibrated
        sqrt(x' cov x + s2): coefficient uncertainty + noise."""
        coef, cov, s2 = post
        anchors = np.atleast_1d(anchors)
        X, _ = self.design(series, anchors, base)
        mean = series["glucose"].to_numpy()[anchors][:, None] + X @ coef
        return mean, self.kappa * np.sqrt(row_variance(X, cov) + s2)

    # The 80 % band (set by train.py on the tuning days): errors after meals are skewed (big spikes run higher
    # than predicted) and heavy-tailed, so instead of mean +- 1.28 sd the band is the 10th and 90th percentile of
    # (truth - prediction) / sd^alpha at each horizon, times sd^alpha. alpha (0..1) sets how much the band follows
    # the model's own uncertainty versus a typical error for that horizon.
    band_q = None
    band_alpha = 1.0

    def band(self, pred, sd, steps):
        """(low, high) of the 80 % band. steps = how many 5-min steps after the last known glucose."""
        if self.band_q is None:
            return pred - 1.2816 * sd, pred + 1.2816 * sd
        h = np.clip(np.asarray(steps) - 1, 0, HORIZON - 1)
        s = np.asarray(sd) ** self.band_alpha
        return pred + self.band_q[0][h] * s, pred + self.band_q[1][h] * s

    def fit_priors(self, people, stats=None):
        """people: list of (series, info). Fits population and group coefficients.
        `stats` = {person id: Stats} already computed (skips rebuilding them)."""
        stats = stats or {i["id"]: self.stats(s, baseline(s)) for s, i in people}
        groups = {i["id"]: i["group"] for _, i in people}
        self.group_stats = {}
        for pid, st in stats.items():
            self.group_stats[groups[pid]] = self.group_stats.get(groups[pid], Stats(self.n_features)) + st
        self.total = sum(self.group_stats.values(), Stats(self.n_features))
        self.person_stats = stats
        return self.set_strengths(self.group_strength, self.person_strength, self.LOOSE_FACTOR)

    def set_strengths(self, group_strength, person_strength, loose_factor):
        """Re-solve the priors for new shrinkage settings (cheap: the data summaries are kept)."""
        self.group_strength, self.person_strength, self.LOOSE_FACTOR = group_strength, person_strength, loose_factor
        loose = np.array([any(t in f for t in self.LOOSE) for f in self.features])
        self.scale = (np.diag(self.total.G) / self.total.n + 1e-9) * np.where(loose, loose_factor, 1.0)  # unit-free
        self.population = solve(self.total, np.zeros(self.n_features), 100.0, self.scale)
        self.groups = {g: solve(st, self.population, group_strength, self.scale) for g, st in self.group_stats.items()}
        self.s2_population = residual_variance(self.total, self.population)
        return self

    def prior(self, group, exclude=None):
        """Group coefficients. `exclude` = a person id (or Stats) to take out of both the population and
        the group first, so they are tested as someone the model never saw."""
        if exclude is None:
            return self.groups.get(group, self.population)
        ex = self.person_stats[exclude] if isinstance(exclude, str) else exclude
        population = solve(self.total - ex, np.zeros(self.n_features), 100.0, self.scale)
        return solve(self.group_stats[group] - ex, population, self.group_strength, self.scale)

    def personalize(self, series, group, base, until=None, prior=None, rows=None):
        """Person coefficients from their own history, pulled toward their group."""
        prior = self.prior(group) if prior is None else prior
        return solve(self.stats(series, base, until, rows), prior, self.person_strength, self.scale)

    def posterior(self, series, group, base, prior=None):
        """(coef, cov, s2) for a person: their coefficients and how uncertain they are."""
        prior = self.prior(group) if prior is None else prior
        if not hasattr(self, "s2_population"):                    # models saved before this was stored
            self.s2_population = (residual_variance(self.total, self.population) if hasattr(self.total, "yy")
                                  else float(np.mean(np.square(getattr(self, "sigma", np.full(HORIZON, 20.0))))))
        return posterior(self.stats(series, base), prior, self.person_strength, self.scale, self.s2_population)

    def components(self, series, anchor, post, base):
        """Split one prediction into named parts, each with its uncertainty.
        Returns (mean, sd): DataFrames, rows = minutes after the anchor (5..240), columns = parts
        (each is its contribution to the change from the anchor glucose)."""
        coef, cov, _ = post
        X, _ = self.design(series, np.array([anchor]), base)
        names = np.array([self.group_of(f) for f in self.features])
        mean, sd = {}, {}
        for name in dict.fromkeys(names):
            j = np.flatnonzero(names == name)
            x = X[0][:, j]
            mean[name] = x @ coef[j]
            sd[name] = self.kappa * np.sqrt(row_variance(x, cov[np.ix_(j, j)]))
        index = np.arange(1, HORIZON + 1) * 5
        return pd.DataFrame(mean, index=index), pd.DataFrame(sd, index=index)


def predict_window(model, series, post, start, end, base=None):
    """Fill in glucose from `start` to `end`, any length, with either model kind. Keeps known glucose up to
    the first gap, then rolls forward 4 h at a time, feeding predictions back in as the next anchor.
    Returns (prediction, sd). The uncertainty of each new anchor is carried into the next 4 h
    (variances add), so long forecasts widen honestly."""
    s = series.copy()
    base = baseline(series) if base is None else base
    times = s.index[(s.index >= start) & (s.index <= end)]
    pos = s.index.get_indexer(times)
    known = s["glucose"].notna().to_numpy()
    missing = pos[~known[pos]]
    var = pd.Series(0.0, index=s.index)
    if len(missing) == 0:
        return s["glucose"].loc[times], var.loc[times]
    before = np.flatnonzero(known[:missing[0]])
    anchor = before[-1] if len(before) else missing[0]
    col = s.columns.get_loc("glucose")
    if np.isnan(s.iloc[anchor, col]):
        s.iloc[anchor, col] = base                       # no glucose at all: start at their normal level
        var.iloc[anchor] = series["glucose"].var() if series["glucose"].notna().sum() > 2 else 30.0 ** 2
    s.iloc[anchor + 1:, col] = np.nan                    # everything after the anchor is unknown
    while anchor < pos[-1]:
        mean, sd = model.predict_with_sd(s, anchor, post, base)
        stop = min(anchor + HORIZON, len(s) - 1)
        s.iloc[anchor + 1: stop + 1, col] = mean[0][: stop - anchor]
        var.iloc[anchor + 1: stop + 1] = var.iloc[anchor] + sd[0][: stop - anchor] ** 2
        anchor = stop
    return s["glucose"].loc[times], np.sqrt(var.loc[times])
