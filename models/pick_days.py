"""Find the whole days the model predicts best for the sample people (choose_demos.py), in the Full Day page's setup.

Setup: glucose and inputs up to midnight, then the day's meals (with their insulin), other insulin doses and the
exercise sessions seen in the recorded activity. Activity, heart rate and basal otherwise follow the person's usual
pattern. The person is taken out of the priors first (never-seen).

Run:  python models/pick_days.py      -> results/day_candidates.csv
"""
import os
os.environ["MKL_THREADING_LAYER"] = "TBB"

import pickle
import warnings

import numpy as np
import pandas as pd

import metabolic
import t1d
from common import ROOT, WEIGHTS, COLUMNS, baseline, predict_window
from plan import NUTRIENTS, plan_series, sessions_from

warnings.filterwarnings("ignore")
DAY = pd.Timedelta(days=1)


def day_plan(series, t0):
    """The day's meals (with the insulin taken with them), other doses and exercise, as the site would get them."""
    day = series[(series.index >= t0) & (series.index < t0 + DAY)]
    meal_rows = day[day["carbs"] > 0]
    meals = [{"time": t, **{c: float(r[c]) for c in NUTRIENTS}, "insulin": float(r["bolus"])} for t, r in meal_rows.iterrows()]
    other = day["bolus"][(day["bolus"] > 0) & (day["carbs"] <= 0)]
    doses = list(zip(other.index, other.to_numpy()))
    return meals, doses, sessions_from(series[series.index < t0 + DAY], t0, t0 + DAY)


def sample_people():
    """The sample people chosen by choose_demos.py: (backend, dataset id)."""
    import json
    chosen = json.loads((ROOT / "results" / "demo_people.json").read_text())
    return [(b, int(i.split("-")[1]) if b == "metabolic" else i) for where, b, i in chosen.values() if where == "try_me"]


def main():
    from pick_demos import best_kind
    with open(WEIGHTS / "models.pkl", "rb") as f:
        bundle = pickle.load(f)
    rows = []
    for backend, pid in sample_people():
        kind = best_kind(bundle, backend)
        series, info = metabolic.load(pid) if backend == "metabolic" else t1d.load(pid)
        m = bundle[(backend, kind)]
        first = series.index[0].normalize()
        for t0 in pd.date_range(first + pd.Timedelta(days=7), series.index[-1].normalize() - DAY, freq="D"):
            truth = series["glucose_raw"][(series.index >= t0) & (series.index < t0 + DAY)]
            hist = series[series.index < t0]
            if truth.notna().mean() < 0.9 or hist["glucose"].iloc[-6:].isna().all():
                continue
            meals, doses, sessions = day_plan(series, t0)
            if len(meals) < 2:
                continue
            b = baseline(hist)
            prior = m.prior(info["group"], exclude=info["id"] if info["id"] in m.person_stats else None)
            post = m.posterior(hist, info["group"], b, prior=prior)
            s = plan_series(hist, t0, t0 + DAY, backend, meals, sessions, doses)
            p, sd = predict_window(m, s[COLUMNS], post, t0, t0 + DAY - pd.Timedelta(minutes=5), b)
            both = pd.concat([p, sd, truth], axis=1, keys=["p", "sd", "a"]).dropna()
            err = both["p"] - both["a"]
            rows.append({"backend": backend, "id": info["id"], "kind": kind, "day": t0.date(), "meals": len(meals),
                         "sessions": len(sessions), "doses": len(doses), "rmse": round(float(np.sqrt(np.mean(err ** 2))), 1),
                         "inside": round(float((np.abs(err) <= 1.2816 * both["sd"]).mean()), 2),
                         "true_peak": round(truth.max()), "pred_peak": round(p.max())})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows).sort_values(["id", "rmse"])
    df.to_csv(ROOT / "results" / "day_candidates.csv", index=False)
    print(df.groupby("id").head(3).to_string(index=False))


if __name__ == "__main__":
    main()
