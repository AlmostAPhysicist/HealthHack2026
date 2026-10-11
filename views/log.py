"""Log: type in glucose, meals, exercise, insulin and a little about yourself, or start from files you already have
(CSV or Excel, any common shape) and fill in what's missing. You get CSV files in the format the Meal and Full Day
pages read (as your history, or as what really happened to check a prediction)."""
import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from style import TYPE_NAME, brand, css

css()
st.markdown("""<style>
  .title {font-size: 3rem; line-height: 1.05; font-weight: 700; margin: 1rem 0 .4rem;}
  .lead {font-size: 1.25rem; color: var(--tx2); max-width: 46rem;}
  .part {font-size: 1.35rem; margin: 1.6rem 0 .3rem;}
</style>""", unsafe_allow_html=True)
brand("2.4rem")
st.markdown("<div class='title'>Log your data</div>"
            "<div class='lead'>No export from your sensor? Type it in (or paste rows from a spreadsheet) and we'll make the "
            "files for you. Every table takes pasted rows: select the first cell and paste.</div>", unsafe_allow_html=True)

INTENSITY = ["light", "moderate", "hard"]
EXERCISE_LOAD = {"metabolic": {"light": 10, "moderate": 25, "hard": 45}, "t1d": {"light": 3, "moderate": 5, "hard": 7}}
today = pd.Timestamp.now().normalize()


def table(key, columns, rows=None, config=None):
    df = pd.DataFrame(rows or [], columns=columns)
    for c in columns:
        if c in ("time", "start"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return st.data_editor(df, num_rows="dynamic", key=f"{key}_{SIG}", width="stretch", column_config=config)


st.markdown("<div class='part c-cyan'>Start from files you have (optional)</div><p class='small'>A glucose export, a "
            "CGMacros file, meal or insulin lists: CSV or Excel. Everything we can read fills the tables below, where you "
            "can fix it and add what's missing (meals, exercise, ...).</p>", unsafe_allow_html=True)
uploads = st.file_uploader("Your files", type=["csv", "txt", "xlsx", "xls"], accept_multiple_files=True, key="log_up",
                           label_visibility="collapsed")
SIG = "-".join(f"{f.name}{f.size}" for f in uploads or []) or "blank"
pre = {"glucose": [], "meals": [], "exercise": [], "doses": [], "profile": {}}
if uploads:
    from engine import csv_rows, meals_in, read, read_meals, read_table, sort_files
    found = sort_files(uploads)
    if found["history"] is not None:
        h = read(found["history"])
        g = h["glucose_raw"].dropna()
        pre["glucose"] = [{"time": t, "glucose": round(float(v), 1)} for t, v in g.items()]
        pre["meals"] = meals_in(h, h.index[0], h.index[-1] + pd.Timedelta(minutes=5))
        b = h["bolus"][(h["bolus"] > 0) & ~(h[["carbs", "protein", "fat", "fiber", "calories"]].fillna(0) > 0).any(axis=1)]
        pre["doses"] = [{"time": t, "units": float(u)} for t, u in b.items()]
    if found["meal"] is not None:
        pre["meals"] += read_meals(found["meal"])
    pre["exercise"] += [r for r in csv_rows(found["exercise"])]
    pre["doses"] += [r for r in csv_rows(found["doses"])]
    if found["profile"] is not None:
        pre["profile"] = read_table(found["profile"]).iloc[0].to_dict()
    st.markdown(f"<p class='small'>Read: {len(pre['glucose'])} glucose readings, {len(pre['meals'])} meals, "
                f"{len(pre['exercise'])} exercise sessions, {len(pre['doses'])} insulin shots"
                + (f"; not recognised: {', '.join(found['unknown'])}" if found["unknown"] else "") + ".</p>",
                unsafe_allow_html=True)


purpose = st.radio("What is this?", ["My history (to predict from)", "What really happened (to check a prediction)"],
                   horizontal=True)
checking = purpose.startswith("What")

st.markdown("<div class='part c-blue'>About you</div>", unsafe_allow_html=True)
a1, a2, a3, a4 = st.columns(4)
prof = pre["profile"]
num = lambda v, d: float(v) if v is not None and str(v).strip() not in ("", "nan") else d
name = a1.text_input("Name", str(prof.get("name", "") or ""), key=f"log_name_{SIG}")
types = ["not sure", "healthy", "prediabetes", "t2d", "t1d"]
kind = a2.selectbox("Diabetes type", types, index=types.index(prof.get("diabetes_type")) if prof.get("diabetes_type") in types else 0,
                    format_func=lambda t: {"not sure": "not sure", **TYPE_NAME}.get(t, t), key=f"log_type_{SIG}",
                    help="About 4 in 10 people with diabetes don't know it. If you know your type, say so.")
a1c = a3.number_input("A1c % (0 if you don't know)", 0.0, 15.0, num(prof.get("a1c"), 0.0), 0.1, key=f"log_a1c_{SIG}")
age = a4.number_input("Age", 0, 120, int(num(prof.get("age"), 0)), key=f"log_age_{SIG}")
t1 = kind == "t1d"

st.markdown("<div class='part c-red'>Glucose readings</div><p class='small'>Date and time, and the reading in mg/dL "
            "(or mmol/L: we convert anything under 35).</p>", unsafe_allow_html=True)
glucose = table("log_glucose", ["time", "glucose"], rows=pre["glucose"], config={
    "time": st.column_config.DatetimeColumn("date and time", format="YYYY-MM-DD HH:mm", step=60),
    "glucose": st.column_config.NumberColumn("glucose", min_value=1.0, max_value=600.0)})
pasted = st.text_area("Or paste readings as text, one per line: 2024-05-01 08:00, 112", height=90, key="log_paste")

st.markdown("<div class='part c-yellow'>Meals</div>", unsafe_allow_html=True)
meal_cols = ["time", "meal", "carbs", "protein", "fat", "fiber", "calories"] + (["insulin"] if t1 else [])
meals = table("log_meals", meal_cols, rows=pre["meals"], config={
    "time": st.column_config.DatetimeColumn("date and time", format="YYYY-MM-DD HH:mm", step=300),
    "meal": st.column_config.SelectboxColumn("meal", options=["Breakfast", "Lunch", "Dinner", "Snack", "Drink", "Other"]),
    **{c: st.column_config.NumberColumn(f"{c} (g)" if c not in ("calories", "insulin") else
                                        ("calories" if c == "calories" else "insulin with it (units)"), min_value=0.0)
       for c in meal_cols[2:]}})

e1, e2 = st.columns(2, gap="large")
with e1:
    st.markdown("<div class='part c-purple'>Exercise</div>", unsafe_allow_html=True)
    exercise = table("log_exercise", ["start", "minutes", "intensity"], rows=pre["exercise"], config={
        "start": st.column_config.DatetimeColumn("starts", format="YYYY-MM-DD HH:mm", step=300),
        "minutes": st.column_config.NumberColumn("minutes", min_value=5, max_value=600, step=5),
        "intensity": st.column_config.SelectboxColumn("how hard", options=INTENSITY)})
with e2:
    st.markdown("<div class='part c-orange'>Insulin shots</div>", unsafe_allow_html=True)
    doses = table("log_doses", ["time", "units"], rows=pre["doses"], config={
        "time": st.column_config.DatetimeColumn("date and time", format="YYYY-MM-DD HH:mm", step=300),
        "units": st.column_config.NumberColumn("units", min_value=0.0, max_value=100.0, step=0.5)})


def readings():
    """Glucose rows from the table and the pasted text."""
    g = glucose.dropna()
    rows = [(pd.Timestamp(t), float(v)) for t, v in zip(g["time"], g["glucose"])]
    for line in pasted.splitlines():
        parts = [p.strip() for p in line.replace(";", ",").replace("\t", ",").split(",") if p.strip()]
        if len(parts) >= 2:
            try:
                rows.append((pd.Timestamp(" ".join(parts[:-1])), float(parts[-1])))
            except (ValueError, TypeError):
                pass
    df = pd.DataFrame(rows, columns=["time", "glucose"]).sort_values("time")
    if len(df) and df["glucose"].median() < 35:
        df["glucose"] *= 18.016
    return df


def build():
    """The files, as {name: csv text}."""
    g = readings()
    m = meals.dropna(subset=["time"])
    m = m[(m[meal_cols[2:7]].fillna(0) > 0).any(axis=1)]
    out = {}
    if checking:
        out["4_what_really_happened.csv"] = g.to_csv(index=False)
        return out
    hist = g.set_index("time")
    for _, r in m.iterrows():
        t = pd.Timestamp(r["time"])
        for c in meal_cols[2:]:
            name_ = "bolus" if c == "insulin" else c
            hist.loc[t, name_] = hist.loc[t, name_] + float(r[c] or 0) if t in hist.index and name_ in hist and pd.notna(hist.loc[t, name_]) else float(r[c] or 0)
    for _, r in doses.dropna().iterrows():
        t = pd.Timestamp(r["time"])
        hist.loc[t, "bolus"] = float(r["units"]) + (hist.loc[t, "bolus"] if t in hist.index and "bolus" in hist and pd.notna(hist.loc[t, "bolus"]) else 0.0)
    load = EXERCISE_LOAD["t1d" if t1 else "metabolic"]
    for _, r in exercise.dropna().iterrows():
        for k in range(max(int(r["minutes"]) // 5, 1)):
            t = pd.Timestamp(r["start"]).floor("5min") + pd.Timedelta(minutes=5 * k)
            hist.loc[t, "activity"] = load[r["intensity"]] if r["intensity"] in load else load["moderate"]
    hist = hist.sort_index().rename_axis("time")
    out["2_my_glucose_history.csv"] = hist.reset_index().to_csv(index=False)
    about = {"name": name, "diabetes_type": kind if kind != "not sure" else "", "a1c": a1c if a1c > 0 else "",
             "age": age if age > 0 else ""}
    out["1_about_me.csv"] = pd.DataFrame([about]).to_csv(index=False)
    return out


files = build()
n_g = len(readings())
st.markdown(f"<p class='muted' style='margin-top:1.2rem'>{n_g} glucose readings, {len(meals.dropna(subset=['time']))} meals, "
            f"{len(exercise.dropna())} exercise sessions, {len(doses.dropna())} insulin shots.</p>", unsafe_allow_html=True)
if n_g < 12:
    st.markdown("<p class='small c-orange'>Add at least an hour of glucose readings (12 at 5-minute steps) for a "
                "useful history.</p>", unsafe_allow_html=True)
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    for fname, text in files.items():
        z.writestr(fname, text)
b1, b2, b3 = st.columns(3)
b1.download_button("Download the files", buf.getvalue(), "glucklich_my_data.zip", "application/zip",
                   icon=":material/download:", width="stretch", disabled=n_g == 0)


def hand_over(page):
    st.session_state["log_files"] = [type("Logged", (io.BytesIO,), {})(text.encode()) for text in files.values()]
    for f, fname in zip(st.session_state["log_files"], files):
        f.name = fname
    st.switch_page(page)


if not checking:
    if b2.button("Use them on the Meal page", width="stretch", disabled=n_g < 12):
        hand_over("views/meal.py")
    if b3.button("Use them on the Full Day page", width="stretch", disabled=n_g < 12):
        hand_over("views/full_day.py")
