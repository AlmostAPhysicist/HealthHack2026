"""Non-meal events: glucose also moves without food.

    exercise sessions   activity well above the person's usual (z > 2 for 15+ min). Split into
                        "after a meal" (meal in the 90 min before / 30 min after the start) and "on its own":
                        exercise after eating usually blunts the meal spike; on an empty stomach it can drop
                        glucose, or raise it briefly when intense (adrenaline).
    quiet time blocks   4-hour blocks with no meal and no exercise (night, dawn, morning, afternoon, evening,
                        late evening). Catches the dawn rise, overnight drift, sleep.

Each event carries context from before it (intensity, duration, glucose vs usual, the person's usual level,
and their own past glucose change after exercise). Both models use these as extra inputs; the overlay model
gives each event type its own spike/cooldown shapes.
"""
import numpy as np
import pandas as pd

PAST = 864                        # steps = 3 days for "usual"
NEAR_BEFORE, NEAR_AFTER = 18, 6   # a meal 90 min before or 30 min after the start = exercise after a meal
BLOCK_HOURS = [0, 4, 8, 12, 16, 20]
BLOCK_NAMES = ["night 0-4", "dawn 4-8", "morning 8-12", "afternoon 12-16", "evening 16-20", "late 20-24"]

EXERCISE_INPUTS = ["ex_meal", "ex_alone", "ex_meal_x_int", "ex_alone_x_int", "ex_x_pre", "ex_expected"]
QUIET_INPUTS = [f"quiet_{h}" for h in BLOCK_HOURS] + ["quiet_x_pre", "quiet_x_level"]
EVENT_INPUTS = EXERCISE_INPUTS + QUIET_INPUTS


def usual(x, steps=PAST):
    """Median of the past (not including now)."""
    return pd.Series(x).rolling(steps, min_periods=36).median().shift(1).to_numpy()


def exercise_table(series):
    """One row per exercise session with its context."""
    act = series["activity"].fillna(0).to_numpy(float)
    g = series["glucose"].to_numpy(float)
    carbs = series["carbs"].fillna(0).to_numpy(float)
    base = pd.Series(act).rolling(PAST, min_periods=36)
    mid, spread = base.median().shift(1).to_numpy(), base.std().shift(1).to_numpy()
    z = (pd.Series(act).rolling(3).mean().to_numpy() - mid) / np.where(spread > 0, spread, np.nan)
    active = np.nan_to_num(z) > 2
    edges = np.flatnonzero(np.diff(np.r_[0, active.astype(int), 0]))
    level = usual(g)
    rows, done = [], []                                   # done = (end step, near_meal, glucose change 1 h after start)
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < 3:
            continue
        near = carbs[max(0, a - NEAR_BEFORE): a + NEAR_AFTER + 1].sum() > 0
        past = [d for end, nm, d in done if end <= a and nm == near]
        rows.append({"step": a, "minutes": (b - a) * 5, "near_meal": near,
                     "intensity": float(np.clip(np.nanmean(z[a:b]), 0, 10)),
                     "pre_vs_normal": g[a] - level[a] if np.isfinite(g[a]) and np.isfinite(level[a]) else 0.0,
                     "past_change": np.mean(past[-10:]) if past else np.nan})
        if a + 12 < len(g) and np.isfinite(g[a]) and np.isfinite(g[a + 12]):
            done.append((a + 12, near, g[a + 12] - g[a]))
    return pd.DataFrame(rows, columns=["step", "minutes", "near_meal", "intensity", "pre_vs_normal", "past_change"])


def quiet_table(series, exercise=None):
    """One row per quiet 4-hour block: no meal in the 2 h before or during it, no exercise during it."""
    g = series["glucose"].to_numpy(float)
    carbs = series["carbs"].fillna(0).to_numpy(float)
    ex = np.zeros(len(g), bool)
    for _, r in (exercise_table(series) if exercise is None else exercise).iterrows():
        ex[int(r["step"]): int(r["step"]) + int(r["minutes"] // 5)] = True
    level = usual(g)
    idx = series.index
    starts = np.flatnonzero((idx.minute == 0) & np.isin(idx.hour, BLOCK_HOURS))
    rows = []
    for a in starts:
        if carbs[max(0, a - 24): a + 48].sum() > 0 or ex[a: a + 48].any():
            continue
        rows.append({"step": a, "hour": idx[a].hour,
                     "pre_vs_normal": g[a] - level[a] if np.isfinite(g[a]) and np.isfinite(level[a]) else 0.0,
                     "usual_level": level[a] if np.isfinite(level[a]) else 100.0})
    return pd.DataFrame(rows, columns=["step", "hour", "pre_vs_normal", "usual_level"])


def event_inputs(series):
    """The extra model inputs, one value per 5-min step (non-zero only at event starts)."""
    n = len(series)
    out = {k: np.zeros(n) for k in EVENT_INPUTS}
    ex = exercise_table(series)
    for _, r in ex.iterrows():
        a, size = int(r["step"]), r["minutes"] / 30           # 30 min of exercise = 1
        kind = "ex_meal" if r["near_meal"] else "ex_alone"
        out[kind][a] += size
        out[f"{kind}_x_int"][a] += size * r["intensity"] / 3
        out["ex_x_pre"][a] += size * r["pre_vs_normal"] / 50
        out["ex_expected"][a] += 0.0 if np.isnan(r["past_change"]) else r["past_change"] / 10
    for _, r in quiet_table(series, ex).iterrows():
        a = int(r["step"])
        out[f"quiet_{int(r['hour'])}"][a] = 1.0
        out["quiet_x_pre"][a] = r["pre_vs_normal"] / 50
        out["quiet_x_level"][a] = (r["usual_level"] - 100) / 50
    return out
