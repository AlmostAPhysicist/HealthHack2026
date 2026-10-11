"""Find the meals the models predict best, for the home page demo and the try_me sample files.

Same setup as the website: the person's glucose and inputs up to the meal, the meal, any insulin doses taken from
30 min before to 3 h after it (the dose plan), and no other meal in those 3 h. The person is taken out of the
population and group priors first, so they are never-seen. Candidates: every CGMacros person and all type 1 test
people, every daytime meal after day 7.

Uses the model the site uses for each backend (the better one on the tuning days).
Run:  python models/pick_demos.py [ids...]      -> results/demo_candidates.csv
"""
import os
os.environ["MKL_THREADING_LAYER"] = "TBB"

import pickle
import warnings

import numpy as np
import pandas as pd

import metabolic
import t1d
from common import ROOT, WEIGHTS, COLUMNS, baseline, posterior, predict_window, split_days

warnings.filterwarnings("ignore")
H3 = pd.Timedelta(hours=3)


def setup(series, t0):
    """History up to the meal; inputs for the 3 h after: the meal and its insulin doses, usual activity/HR/basal."""
    hist = series[series.index < t0]
    s = series[series.index < t0 + H3 + pd.Timedelta(hours=2)].copy()
    after = s.index >= t0
    for c in ["activity", "hr", "basal"]:
        typical = hist[c].groupby(hist.index.time).median()
        s.loc[after, c] = [typical.get(t, np.nan) for t in s.index[after].time]
    s.loc[after & (s.index > t0), ["carbs", "protein", "fat", "fiber", "calories"]] = 0.0
    s.loc[s.index > t0 + H3, "bolus"] = 0.0
    s[["activity", "basal"]] = s[["activity", "basal"]].fillna(0)
    s.loc[after, "glucose"] = np.nan
    return hist, s


def best_kind(bundle, backend):
    """The model the site uses for a backend: lower error on the tuning days (saved by train.py)."""
    return min((k for b, k in bundle if b == backend), key=lambda k: getattr(bundle[(backend, k)], "tune_rmse", np.inf))


def main(only=None):
    with open(WEIGHTS / "models.pkl", "rb") as f:
        bundle = pickle.load(f)
    rows = []
    for backend, people in [("metabolic", metabolic.load_all()), ("t1d", [p for p in t1d.load_all() if t1d.is_test(p[1]["id"])])]:
        for series, info in people:
            if only and info["id"] not in only:
                continue
            first = series.index[0].normalize()
            carbs = series["carbs"].to_numpy()
            for a in np.flatnonzero(carbs > 0):
                t0 = series.index[a]
                if t0 < first + pd.Timedelta(days=7) or not 5 <= t0.hour < 21:
                    continue
                if series["carbs"][(series.index > t0) & (series.index <= t0 + H3)].sum() > 0:
                    continue
                truth = series["glucose_raw"][(series.index >= t0) & (series.index <= t0 + H3)]
                if truth.notna().sum() < 33 or np.isnan(series["glucose"].iloc[a - 1]):
                    continue
                hist, s = setup(series, t0)
                g0 = hist["glucose_raw"].dropna().iloc[-1]
                for kind in (best_kind(bundle, backend),):
                    m = bundle[(backend, kind)]
                    b = baseline(hist)
                    prior = m.prior(info["group"], exclude=info["id"] if info["id"] in m.person_stats else None)
                    post = m.posterior(hist, info["group"], b, prior=prior)
                    p, sd = predict_window(m, s[COLUMNS], post, t0, t0 + H3, b)
                    both = pd.concat([p, sd, truth], axis=1, keys=["p", "sd", "a"]).dropna()
                    err = both["p"] - both["a"]
                    rows.append({"backend": backend, "id": info["id"], "group": info["group"], "kind": kind, "time": t0,
                                 "carbs": carbs[a], "units": round(s["bolus"][(s.index >= t0 - pd.Timedelta(minutes=30)) & (s.index <= t0 + H3)].sum(), 1),
                                 "true_rise": round(truth.max() - g0), "pred_rise": round(p.max() - g0),
                                 "rmse": round(float(np.sqrt(np.mean(err ** 2))), 1), "mape": round(float(np.mean(np.abs(err) / both["a"]) * 100), 1),
                                 "inside": round(float((np.abs(err) <= 1.2816 * both["sd"]).mean()), 2)})
            print(backend, info["id"], len(rows), flush=True)
    df = pd.DataFrame(rows)
    df["rise_error"] = (df["pred_rise"] - df["true_rise"]).abs()
    df.to_csv(ROOT / "results" / "demo_candidates.csv", index=False)
    good = df[(df["true_rise"] >= 30) & (df["carbs"] >= 20)].sort_values(["rmse", "rise_error"])
    for backend in ("metabolic", "t1d"):
        print(f"\n{backend}: best meals (rise >= 30 mg/dL, >= 20 g carbs)")
        print(good[good["backend"] == backend].head(12).to_string(index=False))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
