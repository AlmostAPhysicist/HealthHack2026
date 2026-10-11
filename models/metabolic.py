"""Metabolic backend: people who still make their own insulin (healthy / prediabetes / type 2).
Trained on CGMacros. No administered medication in this data, so the body's own insulin response is
estimated from carbs ("own_insulin" input) instead of read from doses.
"""
import numpy as np
import pandas as pd

from clean import clean
from common import DATA, to_5min
from context import CONTEXT_V2, CONTEXT_V3, CONTEXT_V6, CONTEXT_V7
from events import EVENT_INPUTS, EXERCISE_INPUTS, QUIET_INPUTS

NAME = "metabolic"
GROUPS = ["healthy", "prediabetes", "t2d"]
CGMACROS = DATA / "cgmacros" / "CGMacros_dateshifted365" / "CGMacros"

# Inputs per model version (v1: meal only, v2: + meal context, v3: + calories and full context; see context.py)
REGRESSOR_V1 = ["meal", "carbs", "carbs_am", "carbs_pm", "own_insulin", "protein", "fat", "fiber", "activity", "hr_dev"]
REGRESSOR_V2 = REGRESSOR_V1 + CONTEXT_V2
REGRESSOR_V3 = REGRESSOR_V2 + ["calories"] + CONTEXT_V3
REGRESSOR_V4 = REGRESSOR_V3 + EVENT_INPUTS                           # v4: + exercise and quiet-time events
REGRESSOR_V6 = REGRESSOR_V4 + CONTEXT_V6 + ["protein_late", "fat_late"]   # v6: what else is in the meal
REGRESSOR_V7 = REGRESSOR_V6 + CONTEXT_V7                             # v7: the person's peaks, lows, swings
REGRESSOR_INPUTS = REGRESSOR_V4
# What sets the size of each meal's spikes/cooldowns in the overlay model, per version
OVERLAY_V1 = {"food": ["meal", "carbs", "protein", "fat", "fiber"]}
OVERLAY_V2 = {"food": OVERLAY_V1["food"] + CONTEXT_V2}
OVERLAY_V3 = {"food": OVERLAY_V2["food"] + ["calories"] + CONTEXT_V3}
OVERLAY_V4 = {**OVERLAY_V3, "exercise": EXERCISE_INPUTS, "quiet": QUIET_INPUTS}       # v4
OVERLAY_EVENTS = {**OVERLAY_V4, "food": OVERLAY_V4["food"] + CONTEXT_V6 + CONTEXT_V7}  # v7: meal make-up, peaks, lows


def group_from_a1c(a1c):
    """Standard A1c cutoffs (%)."""
    return "healthy" if a1c < 5.7 else "prediabetes" if a1c < 6.5 else "t2d"


def load_bio():
    """bio.csv: A1c, fasting glucose, BMI, age per subject, plus their group from A1c."""
    bio = pd.read_csv(CGMACROS / "bio.csv")
    bio.columns = bio.columns.str.strip()
    bio = bio.rename(columns={"A1c PDL (Lab)": "a1c", "Fasting GLU - PDL (Lab)": "fasting_glucose",
                              "BMI": "bmi", "Age": "age"})[["subject", "a1c", "fasting_glucose", "bmi", "age"]]
    bio["group"] = [group_from_a1c(a) for a in bio["a1c"]]
    return bio.set_index("subject")


def load(subject):
    """CGMacros subject -> (series, info). Libre glucose; meal macros scaled by % eaten."""
    path = CGMACROS / f"CGMacros-{subject:03d}" / f"CGMacros-{subject:03d}.csv"
    raw = pd.read_csv(path, parse_dates=["Timestamp"]).drop_duplicates("Timestamp").set_index("Timestamp")
    raw.columns = raw.columns.str.strip()
    nothing = pd.Series(np.nan, index=raw.index)
    eaten = raw["Amount Consumed"].fillna(100) / 100 if "Amount Consumed" in raw else 1.0
    df = pd.DataFrame({
        "glucose": raw["Libre GL"], "hr": raw.get("HR", nothing),
        "activity": raw.get("Calories (Activity)", nothing),
        **{c: raw[c.title()] * eaten for c in ["carbs", "protein", "fat", "fiber", "calories"]},
    }, index=raw.index)
    df.loc[raw["Meal Type"].isna(), ["carbs", "protein", "fat", "fiber", "calories"]] = 0   # not a meal row
    bio = load_bio().loc[subject]
    info = {"id": f"cgmacros-{subject:03d}", "source": "CGMacros", "group": bio["group"],
            "a1c": bio["a1c"], "fasting_glucose": bio["fasting_glucose"], "bmi": bio["bmi"], "age": bio["age"]}
    return clean(to_5min(df)), info


def ids():
    return sorted(int(p.name.split("-")[1]) for p in CGMACROS.glob("CGMacros-*"))


def load_all():
    return [load(s) for s in ids()]
