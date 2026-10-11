"""Type 1 backend: people who make no insulin and take it instead (pump, closed loop or injections).
Trained on the MetaboNet parquet: every person whose first 28 days have enough CGM, logged meals and
logged insulin (studies Loop, PEDAP, AZT1D, HUPA-UCM, OhioT1DM). Their bolus (mealtime) and basal
(background) doses are the insulin input; without them glucose after a meal would keep climbing.

Test people are never used to fit the priors: all of OhioT1DM + a fixed 10 % of everyone else.
"""
import zlib


import pandas as pd

from clean import clean
from common import DATA, to_5min
from context import CONTEXT_V2, CONTEXT_V3, CONTEXT_V6, CONTEXT_V7
from events import EVENT_INPUTS, EXERCISE_INPUTS, QUIET_INPUTS

NAME = "t1d"
GROUPS = ["t1d"]
PARQUET = DATA / "Type1Diabetes.parquet"
COHORT = DATA / "cache" / "t1d_cohort.parquet"
STUDIES = ["Loop", "PEDAP", "AZT1D", "HUPA-UCM", "OhioT1DM"]      # studies that log both carbs and insulin
DAYS = 28

# Inputs per model version (v1: meal + insulin only, v2: + meal context, v3: + full context; see context.py).
# No calories: these studies only log carbs.
REGRESSOR_V1 = ["meal", "carbs", "carbs_am", "carbs_pm", "bolus", "basal", "activity", "hr_dev"]
REGRESSOR_V2 = REGRESSOR_V1 + CONTEXT_V2
REGRESSOR_V3 = REGRESSOR_V2 + [c for c in CONTEXT_V3 if c != "carbs_x_kcal_today"]
REGRESSOR_V4 = REGRESSOR_V3 + EVENT_INPUTS                           # v4: + exercise and quiet-time events
REGRESSOR_V6 = REGRESSOR_V4 + CONTEXT_V6 + ["protein_late", "fat_late"]   # v6: what else is in the meal
REGRESSOR_V7 = REGRESSOR_V6 + CONTEXT_V7                             # v7: the person's peaks, lows, swings
REGRESSOR_INPUTS = REGRESSOR_V4
OVERLAY_V1 = {"food": ["meal", "carbs"], "insulin": ["bolus"]}
OVERLAY_V2 = {"food": OVERLAY_V1["food"] + CONTEXT_V2, "insulin": ["bolus"]}
OVERLAY_V3 = {"food": OVERLAY_V2["food"] + [c for c in CONTEXT_V3 if c != "carbs_x_kcal_today"], "insulin": ["bolus"]}
OVERLAY_V4 = {**OVERLAY_V3, "exercise": EXERCISE_INPUTS, "quiet": QUIET_INPUTS}       # v4
OVERLAY_EVENTS = {**OVERLAY_V4, "food": OVERLAY_V4["food"] + CONTEXT_V6 + CONTEXT_V7}  # v7: meal make-up, peaks, lows


def rows():
    """First 28 days of every usable person, pulled from the 1.3 GB parquet once and cached.
    Usable = at least 60 % CGM coverage, 14 logged meals and 14 logged boluses (~380 people)."""
    if not COHORT.exists():
        COHORT.parent.mkdir(exist_ok=True)
        studies = ", ".join(f"'{s}'" for s in STUDIES)
        import duckdb
        duckdb.sql(f"""
            COPY (
              WITH first AS (SELECT source_file, id, MIN(date) d0 FROM '{PARQUET}'
                             WHERE source_file IN ({studies}) GROUP BY 1, 2),
              two_weeks AS (
                SELECT t.source_file || '-' || t.id AS pid, t.source_file AS study, t.date, t.CGM, t.carbs, t.bolus,
                       t.basal, t.heartrate, t.steps, t.age, t.gender, t.insulin_delivery_modality AS modality
                FROM '{PARQUET}' t JOIN first f USING (source_file, id)
                WHERE t.date < f.d0 + INTERVAL {DAYS} DAY),
              usable AS (
                SELECT pid FROM two_weeks GROUP BY pid
                HAVING AVG((CGM IS NOT NULL)::INT) >= 0.6 AND SUM((carbs > 0)::INT) >= 14 AND SUM((bolus > 0)::INT) >= 14)
              SELECT * FROM two_weeks WHERE pid IN (SELECT pid FROM usable) ORDER BY pid, date
            ) TO '{COHORT}' (FORMAT parquet)""")
    return pd.read_parquet(COHORT)


def load(pid, data=None):
    """One person -> (series, info). Basal is a rate: short gaps are carried forward, not read as zero."""
    data = rows() if data is None else data
    raw = data[data["pid"] == pid].set_index("date")
    df = pd.DataFrame({"glucose": raw["CGM"], "carbs": raw["carbs"], "bolus": raw["bolus"],
                       "basal": raw["basal"].ffill(limit=12), "hr": raw["heartrate"],
                       "activity": raw["steps"] / 100}, index=raw.index)
    first = raw.iloc[0]
    info = {"id": pid, "source": first["study"], "group": "t1d", "modality": first["modality"],
            "age": first["age"], "gender": first["gender"]}
    return clean(to_5min(df)), info


def ids(studies=None):
    data = rows()
    pids = data["pid"].unique()
    return sorted(p for p in pids if studies is None or any(p.startswith(s + "-") for s in studies))


def load_all(studies=None):
    """Everyone usable (or only the given studies, e.g. ['OhioT1DM'])."""
    data = rows()
    return [load(p, data) for p in ids(studies)]


def is_test(pid):
    """Held-out test people: all of OhioT1DM + a fixed 10 % of everyone else (by a hash of their id)."""
    return pid.startswith("OhioT1DM") or zlib.crc32(pid.encode()) % 10 == 0


def fill_basal(history, future):
    """Future basal is usually unknown. Repeat the history's typical basal rate for each time of day."""
    if future["basal"].sum() > 0 or history["basal"].sum() == 0:
        return future
    by_time = history["basal"].groupby(history.index.time).median()
    future = future.copy()
    future["basal"] = [by_time.get(t, 0.0) for t in future.index.time]
    return future
