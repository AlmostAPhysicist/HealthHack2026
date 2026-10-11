"""A person's plan for the hours ahead (meals, exercise, insulin) -> the input series the models predict from.

Used by the website and by pick_demos.py, so the demos are picked in exactly the setup the site uses.
After the history ends, activity, heart rate and basal insulin follow the person's usual pattern for that time of
day; planned exercise adds activity on top.
"""
import numpy as np
import pandas as pd

from events import exercise_table

NUTRIENTS = ["carbs", "protein", "fat", "fiber", "calories"]
INTENSITY = ["light", "moderate", "hard"]
EXERCISE_CAP = 30                                                        # minutes of a session that count
EXERCISE_LOAD = {"metabolic": {"light": 10, "moderate": 25, "hard": 45},   # activity added per 5 min (kcal)
                 "t1d": {"light": 3, "moderate": 5, "hard": 7}}          # (steps / 100)


def plan_series(history, start, end, backend, meals=(), sessions=(), doses=()):
    """History before `start`, then the plan up to `end` (+2 h so curves can finish).
    meals: dicts with time and any of carbs, protein, fat, fiber, calories, insulin (units, type 1).
    sessions: (start time, minutes, intensity). doses: (time, units), type 1 only."""
    start = pd.Timestamp(start).floor("5min")
    hist = history[history.index < start]
    grid = pd.date_range(history.index[0], max(pd.Timestamp(end), hist.index[-1]) + pd.Timedelta(hours=2), freq="5min")
    s = hist.reindex(grid)
    new = s.index >= start if len(hist) == 0 else s.index > hist.index[-1]
    for c in NUTRIENTS + ["bolus", "basal", "activity"]:
        s[c] = s[c].fillna(0)
    for c in ["activity", "hr", "basal"]:
        typical = history[c].groupby(history.index.time).median()
        s.loc[new, c] = [typical.get(t, np.nan) for t in s.index[new].time]
    s[["activity", "basal"]] = s[["activity", "basal"]].fillna(0)
    for m in meals:
        t = pd.Timestamp(m["time"]).floor("5min")
        if t not in s.index:
            continue
        for c in NUTRIENTS:
            s.loc[t, c] += float(m.get(c, 0) or 0)
        if backend == "t1d":
            s.loc[t, "bolus"] += float(m.get("insulin", 0) or 0)
    if backend == "t1d":
        for t, units in doses:
            t = pd.Timestamp(t).floor("5min")
            if t in s.index and pd.notna(units):
                s.loc[t, "bolus"] += float(units)
    for t, minutes, intensity in sessions:
        # only a session's first 30 min count: longer ones made the model overreact (tested on 18 never-seen days:
        # average miss 18.7 mg/dL as entered, 18.0 capped at 30 min, 17.6 with no exercise at all)
        span = pd.date_range(pd.Timestamp(t).floor("5min"), periods=max(min(int(minutes), EXERCISE_CAP) // 5, 1), freq="5min")
        s.loc[s.index.isin(span), "activity"] += EXERCISE_LOAD[backend][intensity]
    s.loc[s.index >= start, "glucose"] = np.nan
    return s


def sessions_from(series, start, end):
    """Exercise sessions seen in recorded activity between start and end: (time, minutes, intensity)."""
    ex = exercise_table(series)
    out = []
    for _, r in ex.iterrows():
        t = series.index[int(r["step"])]
        if start <= t < end:
            out.append((t, int(r["minutes"]), "light" if r["intensity"] < 3 else "moderate" if r["intensity"] < 5 else "hard"))
    return out
