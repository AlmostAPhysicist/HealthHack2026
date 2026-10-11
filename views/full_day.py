"""Full Day: drop your history, plan the day (meals, exercise, insulin), predict the whole day, then help us check.

The model only learns from the days before the one it predicts. If the history file already contains that day (a
CGMacros file with day 8 in it, say), its meals, insulin and exercise fill the plan and its glucose is kept back to
check the prediction.
"""
import pandas as pd
import streamlit as st

from engine import (at, DATA_TYPES, PHOTO_TYPES, NUTRIENTS, drop_box, read_readings, TYPE_COLOR, TYPE_NAME, brand, contribute, csv_rows, css, day_insights, day_plot, read_table,
                    exercise_editor, history_plot, insulin_editor, learn, meals_in, predict_plan, read, read_meals, route,
                    accuracy_note, day_headline, track_record,
                    sample_sets, score, sort_files)
from plan import sessions_from

MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snack", "Drink", "Other"]
css()
st.markdown("""<style>
  .title {font-size: 3rem; line-height: 1.05; font-weight: 700; margin: 1rem 0 .4rem;}
  .lead {font-size: 1.25rem; color: var(--tx2); max-width: 44rem;}
  .got {font-size: 1.05rem; line-height: 1.9;}
  .got .ok {color: var(--green);} .got .no {color: var(--tx3);}
  .insight {font-size: 1.2rem; line-height: 1.5; margin: .35rem 0;}
  [data-testid="stFileUploaderDropzone"] {min-height: 8rem; border-width: 2px;}
</style>""", unsafe_allow_html=True)

brand("2.4rem")
st.markdown("<div class='title'>Predict your whole day</div>"
            "<div class='lead'>Tell us about yourself, plan your day (meals, exercise, insulin), and we'll draw your "
            "glucose from morning to midnight, with the highs and lows worth knowing about.</div>", unsafe_allow_html=True)

# ---------- Step 1 ----------
st.markdown("<div class='step-no'>Step 1</div><div class='step-title'>Tell us about yourself</div>"
            "<p class='muted'>Drop your glucose history up to the day (time, glucose, and any of carbs, protein, fat, fiber, "
            "calories, bolus, basal, activity, hr) and an about-me file. If you have your plan as files (meals, exercise, "
            "insulin), drop them too. No data handy? Drag a sample person onto the box.</p>", unsafe_allow_html=True)
sample_sets("day", target="day_files", key="samples_day")
files = drop_box("Drop files", DATA_TYPES, "day_files")
files = (files or []) + st.session_state.get("log_files", [])
found = sort_files(files)
labels = [("history", "glucose history"), ("profile", "about you"), ("meal", "meals for the day"),
          ("exercise", "exercise for the day"), ("doses", "insulin for the day (type 1)")]
if files:
    st.markdown("<div class='got'>" + "<br>".join(
        f"<span class='{'ok' if found[k] is not None else 'no'}'>{'✓' if found[k] is not None else '·'}</span> {label}"
        + (f" <span class='small'>({found[k].name})</span>" if found[k] is not None else "") for k, label in labels)
        + (f"<br><span class='small'>Not recognised: {', '.join(found['unknown'])}</span>" if found["unknown"] else "")
        + "</div>", unsafe_allow_html=True)

profile = read_table(found["profile"]).iloc[0] if found["profile"] is not None else pd.Series(dtype=object)
p1, p2, p3 = st.columns(3)
your_name = p1.text_input("Your name", value=str(profile.get("name", "")) if "name" in profile else "")
types = ["not sure", "healthy", "prediabetes", "t2d", "t1d"]
given = str(profile.get("diabetes_type", "not sure")) if "diabetes_type" in profile else "not sure"
kind = p2.selectbox("Diabetes type", types, index=types.index(given) if given in types else 0,
                    format_func=lambda t: {"not sure": "not sure, work it out", **TYPE_NAME}.get(t, t))
a1c_val = float(profile["a1c"]) if "a1c" in profile and pd.notna(profile["a1c"]) and str(profile["a1c"]).strip() != "" else 0.0
a1c = p3.number_input("A1c % (0 if you don't know)", 0.0, 15.0, a1c_val, 0.1)

if found["history"] is None:
    st.info("Waiting for a glucose history file.")
    st.stop()
try:
    full = read(found["history"])
except Exception as e:
    st.error(f"Couldn't read {found['history'].name}: {e}. It needs a time column and a glucose column.")
    st.stop()
extra = st.session_state.get("day_learned")                     # days validated earlier: we learn from them
if extra is not None:
    full = full.combine_first(extra)
info = {**({"group": kind} if kind != "not sure" else {}), **({"a1c": a1c} if a1c > 0 else {})}
rt = route(full, info)
t1 = rt["backend"] == "t1d"
days = (full.index[-1] - full.index[0]).total_seconds() / 86400
group = rt["group"]
st.markdown(f"<p style='font-size:1.15rem'>{your_name or 'You'}: <span class='c-{TYPE_COLOR[group]}'>{TYPE_NAME[group]}</span>"
            f" <span class='small'>({rt['reason']})</span>. {days:.1f} days of glucose, "
            f"{int((full['carbs'] > 0).sum())} meals logged, usually around {full['glucose'].median():.0f} mg/dL, give or take {full['glucose'].std():.0f} (standard deviation).</p>",
            unsafe_allow_html=True)
history_plot(full, days=2, height=260)

# ---------- Step 2 ----------
st.markdown("<div class='step-no'>Step 2</div><div class='step-title'>Plan your day</div>"
            "<p class='muted'>Add a row for every meal, workout and insulin shot. Rough numbers are fine. Testing us on a "
            "day your file already has? Pick it: we learn only from the days before it and keep that day for checking.</p>",
            unsafe_allow_html=True)
last_glucose = full["glucose_raw"].dropna().index[-1]
file_days = sorted({t.date() for t in full.index[full["glucose_raw"].notna()]})[1:]
after_last = (last_glucose + pd.Timedelta(minutes=5)).date()
logged_ahead = full[(full.index > last_glucose) & (full[NUTRIENTS].fillna(0) > 0).any(axis=1)]
default_day = logged_ahead.index[0].date() if len(logged_ahead) else after_last
options = list(dict.fromkeys(file_days + [after_last, default_day]))
day = st.selectbox("The day", options, index=options.index(default_day),
                   format_func=lambda d: f"{d:%A %d %B %Y}" + (" (in your file: we'll check against it)" if d in file_days else ""))
day0, day_end = pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(hours=23, minutes=55)
history = full[full.index < day0]                                 # the model never sees the day it predicts
src = "-".join(getattr(found[k], "name", "") for k in ("history", "meal", "exercise", "doses")) + f"-{day}"

meal_cols = ["time", "meal", "carbs", "protein", "fat", "fiber", "calories"] + (["insulin"] if t1 else [])
planned = [m for m in read_meals(found["meal"]) if m["time"].date() == day] or meals_in(full, day0, day0 + pd.Timedelta(days=1))
meal_rows = pd.DataFrame(planned).reindex(columns=meal_cols)
meal_rows["time"] = pd.Series([pd.Timestamp(t).time() for t in meal_rows["time"]], dtype=object, index=meal_rows.index)
meal_rows["meal"] = meal_rows["meal"].where(meal_rows["meal"].isin(MEAL_TYPES), "Other") if len(meal_rows) else meal_rows["meal"]
st.markdown("<span class='c-yellow' style='font-size:1.2rem'>Meals</span>", unsafe_allow_html=True)
meal_df = st.data_editor(meal_rows.fillna({c: 0.0 for c in meal_cols[2:]}), num_rows="dynamic", key=f"day_meals_{src}",
                         width="stretch", column_config={
    "time": st.column_config.TimeColumn("time", format="HH:mm", step=300),
    "meal": st.column_config.SelectboxColumn("meal", options=MEAL_TYPES, default="Snack"),
    "carbs": st.column_config.NumberColumn("carbs (g)", min_value=0.0, max_value=1200.0, default=0.0),
    "protein": st.column_config.NumberColumn("protein (g)", min_value=0.0, max_value=900.0, default=0.0),
    "fat": st.column_config.NumberColumn("fat (g)", min_value=0.0, max_value=900.0, default=0.0),
    "fiber": st.column_config.NumberColumn("fiber (g)", min_value=0.0, max_value=300.0, default=0.0),
    "calories": st.column_config.NumberColumn("calories", min_value=0.0, max_value=12000.0, default=0.0),
    "insulin": st.column_config.NumberColumn("insulin with it (units)", min_value=0.0, max_value=50.0, default=0.0)})
meals = [{"time": at(day, r["time"]), "meal": r["meal"],
          **{c: float(r[c]) if pd.notna(r[c]) else 0.0 for c in meal_cols[2:]}}
         for _, r in meal_df.iterrows() if pd.notna(r["time"]) and any((r[c] or 0) > 0 for c in meal_cols[2:7])]
meals.sort(key=lambda m: m["time"])
dropped = sum(1 for _, r in meal_df.iterrows() if pd.notna(r["time"])) - len(meals)
if dropped:
    st.markdown(f"<p class='small c-orange'>{dropped} meal row(s) have no nutrition yet, so they are left out.</p>",
                unsafe_allow_html=True)

x1, x2 = st.columns(2, gap="large")
with x1:
    st.markdown("<span class='c-purple' style='font-size:1.2rem'>Exercise</span>", unsafe_allow_html=True)
    ex_rows = csv_rows(found["exercise"]) or [{"start": t, "minutes": m, "intensity": i} for t, m, i in
                                              sessions_from(full[full.index < day0 + pd.Timedelta(days=1)], day0, day0 + pd.Timedelta(days=1))]
    sessions = exercise_editor(f"day_ex_{src}", day, [r for r in ex_rows if pd.Timestamp(r["start"]).date() == day
                                                      or len(str(r["start"])) <= 8])
with x2:
    if t1:
        st.markdown("<span class='c-orange' style='font-size:1.2rem'>Other insulin shots</span> "
                    "<span class='small'>(not taken with a meal)</span>", unsafe_allow_html=True)
        in_day = full[(full.index >= day0) & (full.index < day0 + pd.Timedelta(days=1))]
        dose_rows = csv_rows(found["doses"]) or [{"time": t, "units": u} for t, u in
                                                 in_day["bolus"][(in_day["bolus"] > 0) & ~(in_day[NUTRIENTS].fillna(0) > 0).any(axis=1)].items()]
        doses = insulin_editor(f"day_doses_{src}", day, dose_rows)
    else:
        doses = []
        st.markdown("<span class='c-orange' style='font-size:1.2rem'>Insulin</span><p class='small'>Insulin doses are used "
                    "for type 1 diabetes. Your own insulin is worked out from what you eat.</p>", unsafe_allow_html=True)

start = max(day0, (history.index[-1] + pd.Timedelta(minutes=5)).floor("5min")) if len(history) else day0
end = day_end
later = full["glucose_raw"][(full.index >= start) & (full.index <= end)].dropna()
go_ = st.button("Predict!", type="primary", disabled=len(history) < 12)
if len(history) < 12:
    st.markdown("<p class='small c-orange'>We need some glucose history before that day.</p>", unsafe_allow_html=True)
sig = str((start, meals, sessions, doses))


def get_person():
    """Learn the person from the days before (only when Predict! is pressed; kept for this session)."""
    with st.spinner(f"Getting to know {your_name or 'you'}: learning how you respond to food..."):
        return learn(history, info, f"day-{found['history'].name}-{kind}-{a1c}-{history.index[-1]}-{len(history)}",
                     kind=str(profile["model"]) if "model" in profile else None,
                     exclude=str(profile["dataset_id"]) if "dataset_id" in profile else None)


if go_:
    person = get_person()
    with st.spinner("Drawing your whole day..."):
        st.session_state.day_pred = (sig, predict_plan(person, history, start, end, meals, sessions, doses),
                                     track_record(person, history))
    st.session_state.pop("day_checked", None)
if "day_pred" not in st.session_state:
    st.stop()
psig, pred, track = st.session_state.day_pred
if psig != sig:
    st.markdown("<p class='c-orange'>You changed the plan. Hit Predict! again.</p>", unsafe_allow_html=True)

truth = st.session_state.get("day_checked")
day_plot(history, pred, meals, sessions, doses, truth)
notes = sorted(day_insights(pred, meals, sessions, doses, rt["backend"]),
               key=lambda n: {"red": 0, "orange": 1, "purple": 2, "yellow": 3}.get(n[0], 9))
notes = [n for n in notes if n[0] in ("red", "orange", "purple", "yellow")][:2]
st.markdown(f"<div style='font-size:1.6rem;margin-top:.6rem'>{day_headline(pred)}</div>"
            + "".join(f"<div style='font-size:1.08rem;line-height:1.55;margin:.25rem 0'><span class='c-{c}'>&#9679;</span> {t}</div>"
                      for c, t in notes)
            + accuracy_note(pred, track), unsafe_allow_html=True)

# ---------- Step 3 ----------
st.markdown("<div class='step-no'>Step 3</div><div class='step-title'>Help us, add what really happened</div>"
            "<p style='font-size:1.15rem'>Help improve our predictions so we can serve more people.</p>",
            unsafe_allow_html=True)
if len(later) >= 12:
    st.markdown(f"<p class='muted'>Your history file already has {len(later)} readings from this day. We hid them from the "
                "model; press Validate! to compare. Or drop another file below.</p>", unsafe_allow_html=True)
else:
    st.markdown("<p class='muted'>Once the day is over, drop your glucose for it: one file, several, a whole folder or a "
                "zip, as CSV, Excel or JSON. We use every reading from that day. Trying a sample? Drag their day onto the "
                "box.</p>", unsafe_allow_html=True)
sample_sets("day_check", target="day_check", key="samples_day_check")
checks = drop_box("Your glucose for the day", DATA_TYPES, "day_check") or ([found["truth"]] if found["truth"] is not None else [])
if st.button("Validate!", type="primary", disabled=not checks and len(later) < 12):
    readings, used = read_readings(checks, pred.index[0], pred.index[-1]) if checks else (None, [])
    st.session_state.day_used = used
    readings = readings if checks else full[full.index >= start]
    readings = readings if readings is not None else pd.DataFrame(columns=["glucose_raw"], index=pd.DatetimeIndex([]))
    got = readings["glucose_raw"][(readings.index >= pred.index[0]) & (readings.index <= pred.index[-1])].dropna()
    st.session_state.day_checked = got
    if len(got):
        learned = readings[(readings.index >= day0) & (readings.index <= day_end)]
        plan = pd.DataFrame([{c: m.get(c, 0.0) for c in NUTRIENTS} for m in meals],
                            index=[m["time"].floor("5min") for m in meals]) if meals else None
        learned = plan.combine_first(learned) if plan is not None else learned
        prev = st.session_state.get("day_learned")
        st.session_state.day_learned = learned if prev is None else prev.combine_first(learned)
        contribute(history, learned, info, "day")
    st.rerun()
if truth is not None:
    s = score(pred, truth)
    if s:
        k1, k2, _ = st.columns(3)
        k1.markdown(f"<div class='small'>This day: typical miss (RMSE)</div><div class='big-number c-green'>{s['rmse']:.0f}</div>"
                    "<div class='small'>mg/dL from the predicted line</div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='small'>Truth inside the band</div><div class='big-number c-green'>{s['inside']:.0%}</div>"
                    f"<div class='small'>of {s['n']} readings</div>", unsafe_allow_html=True)
        if st.session_state.get("day_used"):
            st.markdown("<p class='small'>Checked against " + ", ".join(st.session_state.day_used) + ".</p>",
                        unsafe_allow_html=True)
        st.markdown("<p class='small'>Thank you! The green line in the plot above is what really happened. We've added "
                    "this day to what we know about you, so the next day you predict learns from it, and saved it for the "
                    "next time the model is retrained.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='c-orange'>We found no readings on that day.</p>", unsafe_allow_html=True)

# ---------- Next ----------
st.markdown("<div class='step-no' style='margin-top:3rem'>What next?</div>", unsafe_allow_html=True)
out = pd.DataFrame({"time": pred.index, "predicted_glucose": pred["pred"].round(1),
                    "low_80": pred["lo"].round(1), "high_80": pred["hi"].round(1)})
b1, b2, b3, _ = st.columns([1, 1, 1, 1])
b1.download_button("Download prediction", out.to_csv(index=False), f"glucklich_{day}.csv", "text/csv",
                   icon=":material/download:", width="stretch")
if b2.button("Predict for a meal", width="stretch"):
    st.switch_page("views/meal.py")
if b3.button("Learn more", width="stretch"):
    st.switch_page("views/how_it_works.py")
