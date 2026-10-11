"""Numbers for the How it works page that take too long to compute live.

    average response   mean glucose change in the 3 h after a 30-80 g carb meal (no other meal in those 3 h),
                       for each type: healthy, prediabetes, type 2 (CGMacros), type 1 (MetaboNet training people)
    photo              what the photo model sees on Sam's plate (the home page / How it works example)
    distance corr.     for every meal, how strongly each parameter is related to the rise that followed (peak in
                       the next 2 h minus glucose at the meal). Distance correlation (dcor library) also catches
                       relations that are not straight lines; 0 = unrelated, 1 = fully determined.

Run after photo2nutrition.py:  python models/explain_data.py   -> results/explain.json
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import json

import numpy as np

import metabolic
import t1d
from common import ROOT

STEPS = 37                                    # 0 .. 180 min


def responses(people):
    """Glucose change after each 30-80 g carb meal with a clean 3 h after it."""
    out = []
    for series, info in people:
        g, c = series["glucose_raw"].to_numpy(float), series["carbs"].to_numpy(float)
        for a in np.flatnonzero((c >= 30) & (c <= 80)):
            if a + STEPS > len(g) or c[a + 1: a + STEPS].sum() > 0:
                continue
            w = g[a: a + STEPS]
            if np.isfinite(w).sum() >= STEPS - 3 and np.isfinite(w[0]):
                out.append((info["group"], w - w[0]))
    return out


PARAMETERS = {"carbs": "carbs", "protein": "protein", "fat": "fat", "fiber": "fiber", "bolus": "insulin with the meal",
              "slot": "meal of the day", "hour": "clock time", "carbs_today": "carbs eaten earlier today",
              "mins_since_meal": "time since the last meal", "exercise_2h": "exercise in the 2 h before",
              "pre_vs_normal": "glucose now vs usual", "usual_level": "usual level (3 days)",
              "swing": "swings (3 days)", "swing_all": "swings (all history)", "daily_peak": "usual daily peak",
              "daily_low": "usual daily low", "past_rise_per_10g": "own past rise per 10 g", "trend": "trend (last 30 min)"}


def meal_parameters(people):
    """One row per meal: its parameters and the rise that followed."""
    import pandas as pd
    from context import meal_table
    rows = []
    for series, info in people:
        table = meal_table(series)
        g = series["glucose_raw"].to_numpy(float)
        for _, m in table.iterrows():
            a = int(m["step"])
            after = g[a: a + 25]
            if len(after) < 25 or np.isfinite(after).sum() < 20 or not np.isfinite(g[a]) or a < 6:
                continue
            r = series.iloc[a]
            rows.append({**{k: m[k] for k in PARAMETERS if k in m}, **{k: float(r.get(k, 0) or 0) for k in ("carbs", "protein", "fat", "fiber", "bolus")},
                         "trend": g[a] - g[a - 6], "rise": np.nanmax(after) - g[a]})
    return pd.DataFrame(rows)


def distance_correlations(df):
    import dcor
    out = {}
    for k, label in PARAMETERS.items():
        if k not in df or df[k].nunique() < 3:
            continue
        ok = df[[k, "rise"]].dropna()
        if len(ok) > 2000:
            ok = ok.sample(2000, random_state=0)
        out[label] = round(float(dcor.distance_correlation(ok[k].to_numpy(float), ok["rise"].to_numpy(float))), 3)
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def main():
    meta = metabolic.load_all()
    t1 = [p for p in t1d.load_all() if not t1d.is_test(p[1]["id"])][:150]
    dc = {"metabolic": distance_correlations(meal_parameters(meta)), "t1d": distance_correlations(meal_parameters(t1))}
    print(dc)
    rows = responses(meta) + responses(t1)
    avg = {}
    for group in ("healthy", "prediabetes", "t2d", "t1d"):
        curves = np.array([w for g, w in rows if g == group])
        avg[group] = {"mean": np.round(np.nanmean(curves, axis=0), 1).tolist(),
                      "p25": np.round(np.nanpercentile(curves, 25, axis=0), 1).tolist(),
                      "p75": np.round(np.nanpercentile(curves, 75, axis=0), 1).tolist(), "meals": len(curves)}
        print(group, len(curves), "meals, peak", max(avg[group]["mean"]))
    import photo2nutrition
    res = photo2nutrition.analyse(str(ROOT / "samples" / "home_demo" / "Sam" / "meal_photo.jpg"))
    photo = {"candidates": [(photo2nutrition.food(i), round(s_, 3)) for i, s_ in res["candidates"]],
             "memory": res["memory"], "memory_weight": round(res["memory_weight"], 3), "neighbours": res["neighbours"]}
    print(photo)
    (ROOT / "results" / "explain.json").write_text(json.dumps({"average_response": avg, "photo": photo, "distance_correlation": dc}, indent=1))


if __name__ == "__main__":
    main()
