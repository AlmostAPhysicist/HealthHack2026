"""Exogenous inputs -> effect curves. Turns meals, insulin, activity and HR into the features the regressor uses.

Each input is convolved with a few fixed response shapes (kernels). The regressor then learns how much of
each shape a person has, which gives that person's own response curve per input.

Insulin uses the OpenAPS oref0 exponential insulin curve (open source, MIT), the same math insulin pumps
and dosing calculators use for "insulin on board".
- Type 1: makes no insulin. Their bolus/basal doses go through this curve.
- Everyone else: makes their own insulin in response to carbs. We model that as carbs going through a
  faster version of the same curve ("own_insulin"). Its learned weight is a per-person insulin response.
"""
import numpy as np
from scipy.signal import fftconvolve

from context import CONTEXT_INPUTS, context_inputs
from events import BLOCK_HOURS, EVENT_INPUTS, event_inputs

STEP_MIN = 5


def bump(peak):
    """Rises to 1 at `peak` minutes then fades. Shape of a meal's glucose bump."""
    return lambda t: (t / peak) * np.exp(1 - t / peak)


def ramp(scale):
    """0 -> 1 and stays. A lasting shift (e.g. glucose that is not cleared)."""
    return lambda t: 1 - np.exp(-t / scale)


def insulin_used(peak=75, dia=360):
    """Fraction of an insulin dose that has acted after t minutes (= 1 - insulin on board). OpenAPS oref0.
    Rapid-acting analog defaults: peak 75 min, duration (DIA) 6 h."""
    tau = peak * (1 - peak / dia) / (1 - 2 * peak / dia)
    a = 2 * tau / dia
    S = 1 / (1 - a + (1 + a) * np.exp(-dia / tau))

    def used(t):
        t = np.minimum(t, dia)
        iob = 1 - S * (1 - a) * ((t ** 2 / (tau * dia * (1 - a)) - t / tau - 1) * np.exp(-t / tau) + 1)
        return 1 - iob
    return used


# feature name -> (input, shape)
KERNELS = {name: (col, shape) for name, col, shape in [
    *[(f"meal_b{p}", "meal", bump(p)) for p in (30, 60, 120)],                 # ate something, any size
    *[(f"carbs_b{p}", "carbs", bump(p)) for p in (20, 40, 60, 90, 150)],
    *[(f"carbs_r{p}", "carbs", ramp(p)) for p in (60, 180)],
    *[(f"carbs_am_b{p}", "carbs_am", bump(p)) for p in (40, 100)],             # breakfast vs dinner differ
    *[(f"carbs_pm_b{p}", "carbs_pm", bump(p)) for p in (40, 100)],
    *[(f"carbs_slot{s}_b{p}", f"carbs_slot{s}", bump(p)) for s in (1, 2, 3) for p in (30, 60, 120)],  # meal 1/2/3 of the day
    *[(f"{c}_b{p}", c, bump(p)) for c in CONTEXT_INPUTS if c.startswith("carbs_x_") or c == "carbs_sq" for p in (40, 100)],
    *[(f"calories_b{p}", "calories", bump(p)) for p in (60, 120)],
    # non-meal events (events.py): exercise after a meal / on its own, quiet time-of-day blocks
    *[(f"{c}_b{p}", c, bump(p)) for c in ("ex_meal", "ex_alone") for p in (20, 60, 120)],
    *[(f"{c}_r90", c, ramp(90)) for c in ("ex_meal", "ex_alone")],
    *[(f"{c}_b{p}", c, bump(p)) for c in ("ex_meal_x_int", "ex_alone_x_int", "ex_x_pre", "ex_expected") for p in (30, 90)],
    *[(f"quiet_{h}_b{p}", f"quiet_{h}", bump(p)) for h in BLOCK_HOURS for p in (60, 150)],
    *[(f"quiet_{h}_r240", f"quiet_{h}", ramp(240)) for h in BLOCK_HOURS],
    *[(f"{c}_b{p}", c, bump(p)) for c in ("quiet_x_pre", "quiet_x_level") for p in (60, 180)],
    *[(f"expected_rise_b{p}", "expected_rise", bump(p)) for p in (30, 60, 120)],  # person's own past spikes
    ("own_insulin", "own_insulin", insulin_used(peak=45, dia=240)),            # body's own insulin (non-T1D)
    *[(f"{m}_b{p}", m, bump(p)) for m in ("protein", "fat", "fiber") for p in (60, 150)],
    *[(f"{m}_late_b240", f"{m}_late", bump(240)) for m in ("protein", "fat")],     # v6: the late rise after rich meals
    ("bolus_used", "bolus", insulin_used()),                                   # injected/pumped insulin (T1D)
    ("bolus_active", "bolus", bump(75)),
    ("basal_used", "basal", insulin_used()),
    *[(f"activity_b{p}", "activity", bump(p)) for p in (15, 60)],
    ("activity_r120", "activity", ramp(120)),
    *[(f"hr_b{p}", "hr_dev", bump(p)) for p in (15, 60)],
]}


def kernels_for(inputs):
    """Feature names that use only the given inputs."""
    return [k for k, (col, _) in KERNELS.items() if col in inputs]


def raw_inputs(series):
    """The input signals, one value per 5-min step."""
    cols = ["carbs", "protein", "fat", "fiber", "calories", "bolus", "basal", "activity"]
    s = series.fillna({c: 0 for c in cols})
    hour = s.index.hour + s.index.minute / 60
    carbs = s["carbs"].to_numpy(float)
    return {
        **{c: s[c].to_numpy(float) for c in cols},
        # ate anything: a zero-carb meal (eggs, a shake) is still a meal
        "meal": ((carbs > 0) | (s["calories"].to_numpy(float) > 0) | (s["protein"].to_numpy(float) > 0)
                 | (s["fat"].to_numpy(float) > 0)).astype(float),
        "protein_late": s["protein"].to_numpy(float), "fat_late": s["fat"].to_numpy(float),
        "carbs_am": carbs * ((hour >= 4) & (hour < 11)),
        "carbs_pm": carbs * ((hour >= 17) | (hour < 4)),
        "own_insulin": carbs,
        "hr_dev": (s["hr"] - s["hr"].median()).fillna(0).to_numpy(float),
        **context_inputs(s),
        **event_inputs(s),
    }


def effect_curves(series, kernels):
    """Every input convolved with its shapes over the full past. Returns (n_steps, n_kernels)."""
    inputs = raw_inputs(series)
    t = np.arange(len(series)) * STEP_MIN
    return np.column_stack([fftconvolve(inputs[KERNELS[k][0]], KERNELS[k][1](t))[:len(series)] for k in kernels])


# Readable names for explaining a prediction: feature prefix -> what it means
GROUPS = [
    ("carbs_slot", "meal number of the day"), ("carbs_am", "time of day"), ("carbs_pm", "time of day"),
    ("carbs_x_sin", "time of day"), ("carbs_x_cos", "time of day"),
    ("carbs_x_today", "carbs already eaten today"), ("carbs_x_kcal_today", "calories already eaten today"),
    ("carbs_x_gap", "time since last meal"), ("carbs_x_exercise", "exercise before the meal"),
    ("carbs_x_pre", "glucose before the meal"), ("carbs_x_level", "person's usual glucose"),
    ("carbs_x_swing", "person's glucose swings"), ("carbs_sq", "very large meals"),
    ("carbs_x_fat", "fat and protein in the meal"), ("carbs_x_protein", "fat and protein in the meal"),
    ("carbs_x_fiber", "fiber in the meal"), ("carbs_x_swing_all", "person's glucose swings"),
    ("carbs_x_peak", "person's usual peaks and lows"), ("carbs_x_low", "person's usual peaks and lows"),
    ("expected_rise", "person's past spikes"), ("calories", "calories"),
    ("ex_meal", "exercise after a meal"), ("ex_alone", "exercise on its own"), ("ex_x_pre", "exercise: glucose before"),
    ("ex_expected", "person's past exercise response"), ("quiet_x", "quiet time: glucose vs usual"),
    ("quiet_0", "quiet night 0-4"), ("quiet_4", "quiet dawn 4-8"), ("quiet_8", "quiet morning 8-12"),
    ("quiet_12", "quiet afternoon 12-16"), ("quiet_16", "quiet evening 16-20"), ("quiet_20", "quiet late 20-24"),
    ("meal_", "meal (any size)"), ("carbs_", "carbs"), ("own_insulin", "own insulin response"),
    ("protein", "protein / fat / fiber"), ("fat", "protein / fat / fiber"), ("fiber", "protein / fat / fiber"),
    ("bolus", "insulin doses"), ("basal", "insulin doses"), ("activity", "activity"), ("hr_", "heart rate"),
]


def group_of(feature):
    return next((name for prefix, name in GROUPS if feature.startswith(prefix)), feature)
