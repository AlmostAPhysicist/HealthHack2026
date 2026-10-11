"""Read glucose, meal, insulin and activity CSVs in many shapes into the common 5-min format (common.COLUMNS).

Handles CSV (any separator) and Excel (.xlsx / .xls):
    - our own files (time, glucose, carbs, ...)
    - CGMacros files (Timestamp, Libre GL / Dexcom GL, HR, Calories (Activity), Meal Type, Carbs, ..., Amount Consumed)
    - common meter / app exports: Dexcom Clarity, Libre (LibreView), Nightscout (sgv, dateString), plain "bg"
    - a separate date and time column, day-first or month-first dates, unix seconds / milliseconds
    - glucose in mmol/L (converted to mg/dL)
Anything it can't place is ignored, so a file with extra columns still works.
"""
import numpy as np
import pandas as pd

MMOL = 18.016
# standard name -> accepted header spellings (compared lower-case, without spaces, dashes, dots or brackets)
ALIASES = {
    "time": ["time", "timestamp", "datetime", "date_time", "devicetimestamp", "datestring", "dateandtime",
             "timestampyyyymmddthhmmss", "glucosedisplaytime", "eventtime", "localtime", "created_at", "ts"],
    "glucose": ["glucose", "libregl", "dexcomgl", "sgv", "bg", "cgm", "glucosevaluemgdl", "glucosevalue",
                "historicglucosemgdl", "scanglucosemgdl", "glucosemgdl", "bloodglucose", "sensorglucose", "glucoselevel",
                "historicglucosemmoll", "scanglucosemmoll", "glucosevaluemmoll", "glucosemmoll"],
    "carbs": ["carbs", "carb", "carbohydrate", "carbohydrates", "carbsg", "carbohydratesg", "carbinput", "carbsgrams"],
    "protein": ["protein", "proteins", "proteing"],
    "fat": ["fat", "fats", "fatg", "totalfat"],
    "fiber": ["fiber", "fibre", "fiberg", "dietaryfiber"],
    "calories": ["calories", "kcal", "energy", "energykcal", "caloriesfood", "mealcalories"],
    "bolus": ["bolus", "insulin", "bolusinsulin", "insulinu", "bolusvolumedeliveredu", "insulinunits", "fastactinginsulinunits",
              "rapidactinginsulinunits", "bolusu"],
    "basal": ["basal", "basalrate", "basalinsulin", "basalu", "longactinginsulinunits"],
    "activity": ["activity", "caloriesactivity", "activecalories", "steps", "activitycalories"],
    "hr": ["hr", "heartrate", "heart_rate", "bpm", "pulse"],
}


def key(header):
    return "".join(ch for ch in str(header).strip().lower() if ch.isalnum() or ch == "_").replace("_", "")


def standard(df):
    """Rename recognised columns to the standard names (first match wins). Returns (frame, {standard: original})."""
    found = {}
    by_key = {key(c): c for c in df.columns}
    for name, spellings in ALIASES.items():
        hits = [by_key[k] for k in [key(s) for s in spellings] if k in by_key]
        if name == "glucose" and len(hits) > 1:                 # e.g. CGMacros Libre + Dexcom: keep the fuller one
            hits = sorted(hits, key=lambda c: -pd.to_numeric(df[c], errors="coerce").notna().sum())
        if hits:
            found[name] = hits[0]
    numeric = lambda c: pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5
    if "glucose" not in found:                    # unknown export: a mostly numeric column named like glucose
        c = next((c for c in df.columns if any(w in key(c) for w in ("glucose", "sgv", "cgm"))
                  and "target" not in key(c) and numeric(c)), None)
        if c is not None:
            found["glucose"] = c
    if "time" not in found:                       # and a column named like a time that reads as times
        c = next((c for c in df.columns if any(w in key(c) for w in ("time", "date")) and key(c) not in ("date", "day")
                  and to_times(df[c]).notna().mean() > 0.9), None)
        if c is None and not any(k in by_key for k in ("date", "day")):    # last resort: a text column of date-times
            c = next((c for c in df.columns if not numeric(c) and df[c].astype(str).str.contains(r"\d[-/.]\d").mean() > 0.9
                      and to_times(df[c]).notna().mean() > 0.9), None)
        if c is not None:
            found["time"] = c
    d = next((by_key[k] for k in ("date", "day") if k in by_key), None)
    if "time" in found and d is not None and d != found["time"]:        # a date column plus a clock-only column
        clock = df[found["time"]].astype(str).str.strip()
        if clock.str.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?(\s?[AaPp][Mm])?").mean() > 0.9:
            df = df.assign(_time=df[d].astype(str) + " " + clock)
            found["time"] = "_time"
    if "time" not in found:
        t = next((by_key[k] for k in ("time", "clock", "hourminute") if k in by_key), None)
        if d is not None and t is not None:
            df = df.assign(_time=df[d].astype(str) + " " + df[t].astype(str))
            found["time"] = "_time"
        elif d is not None:
            found["time"] = d
    out = pd.DataFrame({name: df[col] for name, col in found.items()})
    return out, found


def to_times(values):
    """Parse a time column: text in any common order, or unix seconds / milliseconds."""
    v = pd.Series(values)
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().mean() > 0.9:
        unit = "ms" if num.median() > 1e11 else "s"
        return pd.to_datetime(num, unit=unit)
    t = pd.to_datetime(v, errors="coerce", format="mixed")
    if t.notna().mean() < 0.9:
        t = pd.to_datetime(v, errors="coerce", format="mixed", dayfirst=True)
    return t.dt.tz_localize(None) if getattr(t.dt, "tz", None) is not None else t


def read_table(file, nrows=None):
    """A CSV (any separator) or Excel file (.xlsx / .xls) as a raw table."""
    if hasattr(file, "seek"):
        file.seek(0)
    name = str(getattr(file, "name", file)).lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        raw = pd.read_excel(file, nrows=nrows)
    elif name.endswith(".json"):                  # a list of readings, or {"something": [readings]} (e.g. Nightscout)
        import json
        data = json.load(file)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        raw = pd.json_normalize(data)
        raw = raw.head(nrows) if nrows else raw
    else:
        raw = pd.read_csv(file, sep=None, engine="python", nrows=nrows)
    if hasattr(file, "seek"):
        file.seek(0)
    raw.columns = [str(c).strip() for c in raw.columns]
    return raw


def read_frame(file):
    """Any supported CSV or Excel file -> time-indexed frame with the standard columns it has (not yet on the 5-min grid)."""
    raw = read_table(file)
    df, found = standard(raw)
    if "time" not in df:
        raise ValueError("no time column found (looked for time, timestamp, date + time, ...)")
    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    meal_type = next((c for c in raw.columns if key(c) == "mealtype"), None)
    if meal_type is not None:                     # CGMacros: macros count only on meal rows, times the share eaten
        eaten = pd.to_numeric(raw["Amount Consumed"], errors="coerce").fillna(100) / 100 if "Amount Consumed" in raw else 1.0
        for c in ("carbs", "protein", "fat", "fiber", "calories"):
            if c in df:
                df[c] = (df[c] * eaten).where(raw[meal_type].notna(), 0.0)
    df.index = to_times(df.pop("time")).to_numpy()
    df = df[df.index.notna()]
    if "glucose" in df and df["glucose"].notna().any() and df["glucose"].median() < 35:   # mmol/L
        df["glucose"] = df["glucose"] * MMOL
    return df.sort_index()


def has_glucose(file):
    try:
        df = read_frame(file)
        return "glucose" in df and df["glucose"].notna().sum() > 0
    except Exception:
        return False
