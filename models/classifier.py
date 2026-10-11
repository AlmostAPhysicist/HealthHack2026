"""Classify a patient (healthy / prediabetes / t2d / t1d) and pick the backend + weights to use.

Order of trust:
    1. type stated by the user / record
    2. takes mealtime insulin (bolus doses)          -> t1d -> insulin (t1d) backend
    3. lab A1c                                       -> standard cutoffs (5.7 %, 6.5 %)
    4. CGM + activity trends from their history      -> small trained classifier (below)
Then: t1d -> t1d backend, everyone else -> metabolic backend, with their group's prior weights.

The trend classifier is a logistic regression (scikit-learn) on features of the glucose history.
Trained on CGMacros (labels from lab A1c) + OhioT1DM (type 1). Train/test it with:
    python models/classifier.py
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import metabolic
import t1d
from common import WEIGHTS, split_days

PATH = WEIGHTS / "classifier.pkl"
GROUPS = ["healthy", "prediabetes", "t2d", "t1d"]
TREND_FEATURES = ["mean", "std", "cv", "pct_above_140", "pct_above_180", "pct_below_70",
                  "night_mean", "after_meal_rise", "activity_per_day"]


def trends(series):
    """Summary of a glucose history: level, swings, time high/low, overnight level, typical meal rise, activity."""
    g = series["glucose"]
    night = g[g.index.hour < 6]
    rises = []
    vals = g.to_numpy()
    for a in np.flatnonzero(series["carbs"].to_numpy() > 0):
        after = vals[a: a + 25]
        if len(after) == 25 and not np.isnan(after[0]) and np.isfinite(after).any():
            rises.append(np.nanmax(after) - after[0])
    days = max(len(series) / 288, 1)
    return {
        "mean": g.mean(), "std": g.std(), "cv": g.std() / g.mean(),
        "pct_above_140": (g > 140).mean() * 100, "pct_above_180": (g > 180).mean() * 100,
        "pct_below_70": (g < 70).mean() * 100,
        "night_mean": night.mean() if night.notna().any() else g.mean(),
        "after_meal_rise": np.median(rises) if rises else np.nan,
        "activity_per_day": series["activity"].sum() / days,
    }


def gmi(mean_glucose):
    """Glucose Management Indicator: A1c (%) estimated from average CGM glucose (Bergenstal 2018)."""
    return 3.31 + 0.02392 * mean_glucose


class TrendClassifier:
    """Logistic regression on trend features. Missing features -> training median."""

    def fit(self, X, y):
        self.medians = X.median()
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        self.model.fit(X.fillna(self.medians), y)
        return self

    def predict_proba(self, X):
        p = self.model.predict_proba(X[TREND_FEATURES].fillna(self.medians))
        return pd.DataFrame(p, columns=self.model.classes_, index=X.index)


def load_classifier():
    with open(PATH, "rb") as f:
        return pickle.load(f)


def classify(info, series, clf=None):
    """Returns {group, backend, reason, probabilities}."""
    group, reason, probs = None, None, None
    if info.get("group") in GROUPS:
        group, reason = info["group"], "type given"
    elif series["bolus"].sum() > 0:
        group, reason = "t1d", "takes mealtime insulin"
    elif info.get("a1c") is not None and not pd.isna(info.get("a1c")):
        group, reason = metabolic.group_from_a1c(info["a1c"]), f"lab A1c {info['a1c']:.1f}%"
    elif series["glucose"].notna().sum() > 288 / 2:                      # at least half a day of CGM
        clf = clf or load_classifier()
        probs = clf.predict_proba(pd.DataFrame([trends(series)])).iloc[0]
        group, reason = probs.idxmax(), f"CGM trends ({probs.max():.0%} sure)"
    else:
        group, reason = "prediabetes", "not enough data, using the middle group"
    return {"group": group, "backend": "t1d" if group == "t1d" else "metabolic", "reason": reason,
            "probabilities": None if probs is None else probs.round(2).to_dict()}


def main():
    """Leave-one-person-out test on 7-day histories, then save a classifier trained on everyone."""
    people = metabolic.load_all() + t1d.load_all()
    X = pd.DataFrame([trends(split_days(s)[0]) for s, _ in people])
    y = pd.Series([i["group"] for _, i in people])
    pred = []
    for k in range(len(people)):
        train = X.index != k
        pred.append(TrendClassifier().fit(X[train], y[train]).predict_proba(X.iloc[[k]]).iloc[0].idxmax())
    pred = pd.Series(pred)
    gmi_pred = pd.Series([metabolic.group_from_a1c(gmi(m)) for m in X["mean"]])
    print(f"Leave-one-person-out accuracy (7 days of CGM, no labs, no insulin data): {(pred == y).mean():.0%}")
    print(f"Simple rule (A1c estimated from mean glucose) accuracy:              {(gmi_pred == y).mean():.0%}")
    print(pd.crosstab(y, pred, rownames=["true"], colnames=["predicted"]))
    WEIGHTS.mkdir(exist_ok=True)
    with open(PATH, "wb") as f:
        pickle.dump(TrendClassifier().fit(X, y), f)
    print(f"Saved {PATH}")


if __name__ == "__main__":
    import classifier            # run via the module so the saved classifier can be loaded from other files
    classifier.main()
