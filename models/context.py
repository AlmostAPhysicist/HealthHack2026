"""Meal context: what we know about each meal from the day so far and from the person's past.
Every value uses only information from before the meal.

For each meal (a step with carbs > 0):
    slot              which meal of the day it is (1st, 2nd, 3rd or later). A day starts at 4 am.
    hour              clock time of the meal
    carbs_today       carbs already eaten earlier that day
    kcal_today        calories already eaten earlier that day
    mins_since_meal   minutes since the previous meal (capped at 12 h)
    exercise_2h       activity in the 2 h before the meal, above the person's usual (past 3 days)
    pre_vs_normal     glucose at the meal minus the person's usual level (past 3 days)
    usual_level       the person's usual glucose (past 3 days): stands in for medical history (A1c-like)
    swing             how much their glucose varies (std, past 3 days): their sensitivity
    swing_all         the same over their whole history so far (v7)
    daily_peak / low  their typical daily highest / lowest glucose (past 3 days) above / below their usual (v7)
    past_rise_per_10g this person's average spike per 10 g carbs for this meal slot, from earlier meals
                      whose 2 h window is already over (falls back to all slots, then to nothing)

These become extra model inputs: carbs split by slot, and carbs scaled by each context value, so the
models learn e.g. "dinner after a big lunch spikes less", "after exercise spikes less",
"people who run high spike more".
"""
import numpy as np
import pandas as pd

DAY_START = 4                     # hours: meals after midnight count toward the previous day
WINDOW = 24                       # steps = 2 h after a meal
PAST = 864                        # steps = 3 days used for "usual" level, swing and activity

# v2 context (meal number, day so far, exercise, pre-meal glucose, own past spikes)
CONTEXT_V2 = ["carbs_slot1", "carbs_slot2", "carbs_slot3", "carbs_x_today",
              "carbs_x_exercise", "carbs_x_pre", "expected_rise"]
# v3 adds: clock time, calories so far, time since last meal, usual level, swings, very large meals
CONTEXT_V3 = ["carbs_x_kcal_today", "carbs_x_gap", "carbs_x_sin", "carbs_x_cos",
              "carbs_x_level", "carbs_x_swing", "carbs_sq"]
# v6 adds what is in the meal besides carbs: fat and protein slow carb absorption, fiber blunts the spike
CONTEXT_V6 = ["carbs_x_fat", "carbs_x_protein", "carbs_x_fiber"]
# v7 adds more about the person's recent glucose: swings over their whole history, and their typical daily peak
# and low (past 3 days) relative to their usual level
CONTEXT_V7 = ["carbs_x_swing_all", "carbs_x_peak", "carbs_x_low"]
CONTEXT_INPUTS = CONTEXT_V2 + CONTEXT_V3 + CONTEXT_V6 + CONTEXT_V7


def daily(values, fn):
    """Average over whole days (288 steps) of fn (max / min) of the values, ignoring days with little data."""
    days = [values[i: i + 288] for i in range(len(values) % 288, len(values), 288)]
    got = [fn(d) for d in days if np.isfinite(d).sum() > 144]
    return float(np.mean(got)) if got else np.nan


def meal_table(series):
    """One row per meal with its context."""
    carbs = series["carbs"].to_numpy(float)
    kcal = series["calories"].to_numpy(float)
    act = series["activity"].fillna(0).to_numpy(float)
    g = series["glucose"].to_numpy(float)
    day = (series.index - pd.Timedelta(hours=DAY_START)).normalize()
    hours = series.index.hour + series.index.minute / 60
    steps = np.flatnonzero(carbs > 0)
    rows, done = [], []                           # done = (end step, slot, rise per 10 g) of finished meals
    for k, a in enumerate(steps):
        earlier = steps[:k][day[steps[:k]] == day[a]]
        slot = min(len(earlier), 2) + 1
        recent = g[max(0, a - PAST): a]
        seen = np.isfinite(recent).sum() > 12
        usual = np.nanmedian(recent) if seen else np.nan
        past = [r for end, s, r in done if end <= a]
        same = [r for end, s, r in done if end <= a and s == slot]
        rows.append({
            "step": a, "slot": slot, "hour": hours[a],
            "carbs_today": carbs[earlier].sum(), "kcal_today": kcal[earlier].sum(),
            "mins_since_meal": min((a - steps[k - 1]) * 5, 720) if k else 720,
            "exercise_2h": act[max(0, a - 24): a].sum() - (np.median(act[max(0, a - PAST): a]) * 24 if a else 0.0),
            "pre_vs_normal": g[a] - usual if np.isfinite(g[a]) and np.isfinite(usual) else 0.0,
            "usual_level": usual if seen else np.nan,
            "swing": np.nanstd(recent) if seen else np.nan,
            "swing_all": np.nanstd(g[:a]) if np.isfinite(g[:a]).sum() > 12 else np.nan,
            "daily_peak": daily(recent, np.nanmax) - usual if seen else np.nan,
            "daily_low": usual - daily(recent, np.nanmin) if seen else np.nan,
            "past_rise_per_10g": np.mean(same[-10:]) if same else (np.mean(past[-10:]) if past else np.nan),
        })
        after = g[a: a + WINDOW + 1]
        if len(after) == WINDOW + 1 and np.isfinite(after).all():
            done.append((a + WINDOW, slot, (after.max() - after[0]) / max(carbs[a], 5) * 10))
    return pd.DataFrame(rows)


def context_inputs(series, table=None):
    """The extra model inputs, one value per 5-min step (non-zero only at meals)."""
    n = len(series)
    out = {k: np.zeros(n) for k in CONTEXT_INPUTS}
    meals = meal_table(series) if table is None else table
    if meals.empty:
        return out
    a = meals["step"].to_numpy()
    c = series["carbs"].to_numpy(float)[a]
    m = {k: np.nan_to_num(meals[k].to_numpy(float)) for k in meals}
    for s in (1, 2, 3):
        out[f"carbs_slot{s}"][a] = c * (m["slot"] == s)
    out["carbs_x_today"][a] = c * m["carbs_today"] / 100
    out["carbs_x_exercise"][a] = c * m["exercise_2h"] / 100
    out["carbs_x_pre"][a] = c * m["pre_vs_normal"] / 50
    out["expected_rise"][a] = m["past_rise_per_10g"] * c / 10
    out["carbs_x_kcal_today"][a] = c * m["kcal_today"] / 1000
    out["carbs_x_gap"][a] = c * m["mins_since_meal"] / 240
    out["carbs_x_sin"][a] = c * np.sin(2 * np.pi * m["hour"] / 24)
    out["carbs_x_cos"][a] = c * np.cos(2 * np.pi * m["hour"] / 24)
    out["carbs_x_level"][a] = c * (np.where(np.isnan(meals["usual_level"]), 100, meals["usual_level"]) - 100) / 50
    out["carbs_x_swing"][a] = c * np.where(np.isnan(meals["swing"]), 20, meals["swing"]) / 20
    out["carbs_sq"][a] = c ** 2 / 100
    food = series.fillna({k: 0 for k in ("fat", "protein", "fiber")})
    fat, protein, fiber = (food[k].to_numpy(float)[a] for k in ("fat", "protein", "fiber"))
    total = c + fat + protein + 1
    out["carbs_x_fat"][a] = c * fat / total
    out["carbs_x_protein"][a] = c * protein / total
    out["carbs_x_fiber"][a] = c * np.minimum(fiber, 30) / 10
    out["carbs_x_swing_all"][a] = c * np.where(np.isnan(meals["swing_all"]), 20, meals["swing_all"]) / 20
    out["carbs_x_peak"][a] = c * np.where(np.isnan(meals["daily_peak"]), 50, meals["daily_peak"]) / 50
    out["carbs_x_low"][a] = c * np.where(np.isnan(meals["daily_low"]), 20, meals["daily_low"]) / 20
    return out


def warp_features(series, table=None):
    """Per-meal descriptors that set how fast its spike/cooldown play out (overlay model time-warp).
    All roughly in -1..1. Returns (steps, Z)."""
    meals = meal_table(series) if table is None else table
    if meals.empty:
        return np.array([], int), np.zeros((0, len(WARP_FEATURES)))
    a = meals["step"].to_numpy()
    s = series.fillna(0)
    carbs, fat, protein, fiber = (s[c].to_numpy(float)[a] for c in ("carbs", "fat", "protein", "fiber"))
    total = carbs + fat + protein + 1
    Z = np.column_stack([
        np.log1p(carbs) / 4 - 1,                     # meal size
        fat / total, protein / total, fiber / 10,     # what slows absorption
        np.sin(2 * np.pi * meals["hour"] / 24), np.cos(2 * np.pi * meals["hour"] / 24),   # time of day
        meals["mins_since_meal"].to_numpy() / 360 - 1,
        np.clip(np.nan_to_num(meals["exercise_2h"].to_numpy()) / 200, -1, 1),
    ])
    return a, Z


WARP_FEATURES = ["meal size", "fat share", "protein share", "fiber", "time of day (sin)", "time of day (cos)",
                 "time since last meal", "exercise before"]
