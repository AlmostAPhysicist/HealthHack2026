"""Meal: drop your files, tell us your meal (and any exercise or insulin), predict, then help us check.

The model only ever learns from glucose before the meal. If the history file already runs past the meal (a CGMacros
file, say, with day 8 in it), the readings after the meal are hidden from the model and used to check it instead.
"""
import io

import numpy as np
import pandas as pd
import streamlit as st
from style import photo_file
from PIL import Image

from engine import (DATA_TYPES, PHOTO_TYPES, NUTRIENTS, drop_box, read_readings, TYPE_COLOR, TYPE_NAME, brand, contribute, read_table, csv_rows, css, exercise_editor, history_plot,
                    insulin_editor, learn, meal_plot, meals_in, photo_model, predict_after_meal, read, read_meals, rise_text,
                    route, sample_sets, score, sort_files, track_record,
                    accuracy_note, commentary_html, image_picker, meal_commentary,
                    nutrition_line, option_picker)

css()
st.markdown("""<style>
  .title {font-size: 3rem; line-height: 1.05; font-weight: 700; margin: 1rem 0 .4rem;}
  .lead {font-size: 1.25rem; color: var(--tx2); max-width: 44rem;}
  .got {font-size: 1.05rem; line-height: 1.9;}
  .got .ok {color: var(--green);} .got .no {color: var(--tx3);}
  .rise {font-size: 1.6rem; margin-top: .6rem;}
  [data-testid="stFileUploaderDropzone"] {min-height: 8rem; border-width: 2px;}
</style>""", unsafe_allow_html=True)

brand("2.4rem")
st.markdown("<div class='title'>Predict your own glucose</div>"
            "<div class='lead'>Three steps. Drop in your data, tell us what you're about to eat, and we'll draw the next "
            "three hours. Then, if you can, help us check how we did!</div>", unsafe_allow_html=True)

# ---------- Step 1 ----------
st.markdown("<div class='step-no'>Step 1</div><div class='step-title'>Tell us about yourself</div>"
            "<p class='muted'>Drop your files in the box, all at once is fine: your glucose history (our format, a CGMacros "
            "file, or a Dexcom / Libre / Nightscout export), an about-me file (diabetes type, A1c), your meal and its photo. "
            "We sort them out. No data handy? Drag a sample person onto the box, or make your files on the Log page.</p>",
            unsafe_allow_html=True)
sample_sets("meal", target="meal_files", key="samples_meal")
files = drop_box("Drop files", DATA_TYPES + PHOTO_TYPES, "meal_files")
files = (files or []) + st.session_state.get("log_files", [])
found = sort_files(files)
labels = [("history", "glucose history"), ("profile", "about you"), ("meal", "your meal"), ("photo", "meal photo"),
          ("doses", "insulin plan (type 1)"), ("exercise", "exercise")]
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
                    format_func=lambda t: {"not sure": "not sure, work it out", **TYPE_NAME}.get(t, t),
                    help="If you know it, say so: about 4 in 10 people with diabetes don't know they have it, so when you "
                         "don't tell us we work it out from your glucose.")
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
extra = st.session_state.get("meal_learned")                    # readings from earlier validations (we learn from them)
if extra is not None:
    full = full.combine_first(extra)
info = {**({"group": kind} if kind != "not sure" else {}), **({"a1c": a1c} if a1c > 0 else {})}
rt = route(full, info)
days = (full.index[-1] - full.index[0]).total_seconds() / 86400
group = rt["group"]
st.markdown(f"<p style='font-size:1.15rem'>{your_name or 'You'}: <span class='c-{TYPE_COLOR[group]}'>{TYPE_NAME[group]}</span>"
            f" <span class='small'>({rt['reason']})</span>. {days:.1f} days of glucose, "
            f"{int((full['carbs'] > 0).sum())} meals logged, usually around {full['glucose'].median():.0f} mg/dL, give or take {full['glucose'].std():.0f} (standard deviation).</p>",
            unsafe_allow_html=True)
history_plot(full, days=2, height=260)

# ---------- Step 2 ----------
st.markdown("<div class='step-no'>Step 2</div><div class='step-title'>What are you eating?</div>"
            "<p class='muted'>Add each thing you're eating with ＋: a photo, a search, or the numbers.</p>", unsafe_allow_html=True)
NUTR = NUTRIENTS
last_glucose = full["glucose_raw"].dropna().index[-1]
file_meals = read_meals(found["meal"])
in_file = meals_in(full, full.index[0] + pd.Timedelta(days=1), full.index[-1] + pd.Timedelta(minutes=5))
in_file = [m for m in in_file if full["glucose_raw"][(full.index > m["time"]) & (full.index <= m["time"] + pd.Timedelta(hours=2))].notna().sum() >= 12]
choices = (["your meal file"] if file_meals else []) + ["a new meal"] + [f"{m['time']:%a %d %b %H:%M} {m['meal']}, {m['carbs']:.0f} g carbs" for m in in_file[::-1][:30]]
pick = st.selectbox("Predict for", choices, help="Pick a meal that is already in your history file to test us on it: we "
                    "hide everything from that meal on, and use it to check the prediction.") if len(choices) > 1 else choices[0]
if pick == "your meal file":
    chosen = file_meals[0]
elif pick == "a new meal":
    chosen = {"time": (last_glucose + pd.Timedelta(minutes=30)).floor("5min"), "meal": "Your meal", **{c: 0.0 for c in NUTR}, "insulin": 0.0}
else:
    chosen = in_file[::-1][choices.index(pick) - (2 if file_meals else 1)]

src = f"{pick}-{getattr(found['meal'], 'name', '')}-{found['history'].name}-{getattr(found['doses'], 'name', '')}-{getattr(found['exercise'], 'name', '')}"
SHOW = (("carbs", "yellow", " g carbs"), ("protein", "red", " g protein"), ("fat", "orange", " g fat"),
        ("fiber", "green", " g fiber"), ("calories", "purple", " kcal"))
SLIDERS = (("carbs", "Carbs (g)", "yellow", 300), ("protein", "Protein (g)", "red", 200), ("fat", "Fat (g)", "orange", 200),
           ("fiber", "Fiber (g)", "green", 60), ("calories", "Calories", "purple", 3000))
TURN = {"yellow": 195, "red": 156, "orange": 172, "green": 228, "purple": 48}    # hue shift from the theme's blue
line_of = lambda n, k=1.0: " · ".join(f"<span class='c-{c}'>{n.get(f, 0) * k:.0f}{u}</span>" for f, c, u in SHOW)
st.markdown("<style>" + "".join(                                     # every card's nutrient sliders in their colours
    f"[class*='st-key-c'][class*='_{f}'] [role='group'] > div > div:first-child {{filter: hue-rotate({TURN[c]}deg);}}"
    f"[class*='st-key-c'][class*='_{f}'] div[style*='translate(-50%'] {{background-color: var(--{c}) !important;"
    f" border-color: var(--{c}) !important;}}"
    f"[class*='st-key-c'][class*='_{f}'] [data-testid='stSliderThumbValue'] p, [class*='st-key-c'][class*='_{f}'] label p "
    f"{{color: var(--{c}) !important;}}" for f, _, c, _ in SLIDERS) + "</style>", unsafe_allow_html=True)


def set_card(i, nutr):
    """Queue numbers for a card's sliders (applied before they are drawn on the next run)."""
    st.session_state.setdefault("pending", {})[i] = {f: float(round(float(nutr.get(f, 0) or 0))) for f in NUTR}


def add_card(label, nutr, thumb=None, res=None):
    i = st.session_state.plate_next
    st.session_state.plate_next += 1
    st.session_state.plate.append({"id": i, "label": label, "thumb": thumb, "res": res})
    set_card(i, nutr)
    st.session_state.adding = False


def read_photo(data):
    """Look at a photo once (by its bytes); keep the analysis."""
    import hashlib
    k = hashlib.sha1(data).hexdigest()[:12]
    if k not in st.session_state.plate_photos:
        with st.spinner("Looking at the photo..."):
            st.session_state.plate_photos[k] = photo_model().analyse(Image.open(io.BytesIO(data)).convert("RGB"))
    return st.session_state.plate_photos[k]


def choices(res):
    """The 6 ways to set a card from a photo: CLIP's 3 food matches (USDA values) and the 3 closest logged meals."""
    pm = photo_model()
    foods = [(pm.food(i)["name"], pm.plate([(i, 1.0)])) for i, _ in res["candidates"][:3]]
    near = [dict(n, photo=f) for n in res.get("neighbours", []) if (f := photo_file(n.get("photo")))][:3]
    return foods, near


if st.session_state.get("meal_src") != src:                      # new files or a new pick: start the meal from them
    st.session_state.meal_src = src
    st.session_state.meal_date, st.session_state.meal_time = chosen["time"].date(), chosen["time"].time()
    st.session_state.meal_insulin = round(float(chosen.get("insulin", 0) or 0), 1)
    st.session_state.plate, st.session_state.plate_photos, st.session_state.plate_next = [], {}, 0
    st.session_state.adding = False
    for k in ("meal_pred", "meal_checked"):
        st.session_state.pop(k, None)
    if any(float(chosen.get(f, 0) or 0) > 0 for f in NUTR):
        add_card(f"{chosen.get('meal', 'Meal')} from your file", chosen,
                 thumb=found["photo"].getvalue() if found["photo"] is not None and pick == "your meal file" else None)


def card(it):
    """One food: its photo (with 6 choices to set it from), and its own five sliders."""
    i = it["id"]
    with st.container(border=True):
        head, x = st.columns([14, 1], vertical_alignment="center")
        head.markdown(f"**{it['label']}**")
        if x.button("✕", key=f"drop_{i}", help="Remove"):
            st.session_state.plate.remove(it)
            st.rerun()
        c_photo, c_sliders, c_choices = st.columns([1, 1.6, 1.8], gap="large")
        if it["thumb"] is not None:
            c_photo.image(it["thumb"], width="stretch")
            if it["res"] is None and c_photo.button("Get nutrition from photo", key=f"fromphoto_{i}", width="stretch"):
                it["res"] = read_photo(it["thumb"])
                st.rerun()
        for f, v in st.session_state.get("pending", {}).pop(i, {}).items():
            st.session_state[f"c{i}_{f}"] = v
        with c_sliders:
            for f, label, _, top in SLIDERS:
                v = st.session_state.get(f"c{i}_{f}", 0.0)
                st.slider(label, 0.0, float(max(top, 1.5 * v)), step=1.0 if f != "calories" else 10.0, format="%.0f",
                          key=f"c{i}_{f}")
        if it["res"] is not None:
            foods, near = choices(it["res"])
            with c_choices:
                st.markdown("<div class='small'>Set from what CLIP sees (USDA, usual portion):</div>", unsafe_allow_html=True)
                c = option_picker([(name, nutrition_line(n), False) for name, n in foods], key=f"opt_{i}")
                if c is not None:
                    set_card(i, foods[c][1])
                    st.rerun()
                if near:
                    st.markdown("<div class='small' style='margin-top:.4rem'>Or from a logged meal that looks the same:</div>",
                                unsafe_allow_html=True)
                    c = image_picker([(n["photo"], nutrition_line(n), False) for n in near], key=f"near_{i}")
                    if c is not None:
                        set_card(i, near[c])
                        st.rerun()


def add_panel():
    """The + panel: a photo, a search, or blank sliders to set yourself."""
    with st.container(border=True):
        head, x = st.columns([14, 1], vertical_alignment="center")
        head.markdown("**Add a food**")
        if x.button("✕", key="add_close", help="Cancel"):
            st.session_state.adding = False
            st.rerun()
        how = st.radio("How", ["Photo", "Search the food database", "Set it myself"], horizontal=True, key="add_how",
                       label_visibility="collapsed")
        if how == "Photo":
            up = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"], key=f"add_photo_{st.session_state.plate_next}",
                                  label_visibility="collapsed")
            if up is not None:
                data = up.getvalue()
                res = read_photo(data)
                foods, _ = choices(res)
                start = photo_model().plate([(res["candidates"][0][0], 1.0)], res["memory"], res["memory_weight"])
                add_card(foods[0][0], start, thumb=data, res=res)       # our best estimate; the 6 choices are on the card
                st.rerun()
        elif how == "Search the food database":
            names = photo_model().food_names()
            v = st.selectbox("Search", range(len(names)), index=None, key="add_search", placeholder="Type to search 5,200 foods",
                             format_func=lambda o: names[o], label_visibility="collapsed")
            if v is not None:
                add_card(names[v], photo_model().plate([(v, 1.0)]))
                st.rerun()
        else:
            label = st.text_input("Food", placeholder="e.g. apple", key="add_label")
            if st.button("Add", type="primary", key="add_manual_go"):
                add_card(label or "Your food", {f: 0.0 for f in NUTR})
                st.rerun()


d1, d2, d3 = st.columns([1, 1, 1.2])
meal_date = d1.date_input("When", key="meal_date")
meal_clock = d2.time_input("Time", key="meal_time", step=300)
for it in list(st.session_state.plate):
    card(it)
if st.session_state.adding:
    add_panel()
elif st.button("＋", key="add_open", help="Add a food: a photo, a search, or set it yourself"):
    st.session_state.adding = True
    st.rerun()
vals = {f: float(sum(st.session_state.get(f"c{it['id']}_{f}", 0.0) for it in st.session_state.plate)) for f in NUTR}
if st.session_state.plate:
    st.markdown(f"<div style='font-size:1.15rem;margin:.4rem 0'>The meal: {line_of(vals)}</div>", unsafe_allow_html=True)
else:
    st.markdown("<p class='muted'>Press ＋ to add what you're eating.</p>", unsafe_allow_html=True)
if rt["backend"] == "t1d":
    vals["insulin"] = d3.number_input("Insulin with the meal (units)", 0.0, 50.0, step=0.5, key="meal_insulin")

meal_time = pd.Timestamp.combine(meal_date, meal_clock)
history = full[full.index < meal_time]                            # the model never sees anything from the meal on
later = full["glucose_raw"][(full.index >= meal_time) & (full.index <= meal_time + pd.Timedelta(hours=3))].dropna()

st.markdown("<div style='font-size:1.25rem;margin-top:1rem'>Anything else in the next three hours?</div>", unsafe_allow_html=True)
x1, x2 = st.columns(2, gap="large")
with x1:
    st.markdown("<span class='c-purple'>Exercise</span> <span class='small'>(add a row: start, minutes, how hard)</span>",
                unsafe_allow_html=True)
    sessions = exercise_editor(f"meal_ex_{src}", meal_date, csv_rows(found["exercise"]))
with x2:
    if rt["backend"] == "t1d":
        st.markdown("<span class='c-orange'>More insulin</span> <span class='small'>(time, units)</span>", unsafe_allow_html=True)
        planned = csv_rows(found["doses"]) or [{"time": t, "units": u} for t, u in
                                               full["bolus"][(full.index > meal_time) & (full.index <= meal_time + pd.Timedelta(hours=3))
                                                             & (full["bolus"] > 0)].items()]
        doses = insulin_editor(f"meal_doses_{src}", meal_date, planned)
    else:
        doses = []
        st.markdown("<span class='c-orange'>Insulin</span><p class='small'>Insulin doses are used for type 1 diabetes. "
                    "Your own insulin is worked out from what you eat.</p>", unsafe_allow_html=True)
empty = all(vals[c] <= 0 for c in NUTR)
go_ = st.button("Predict!", type="primary", disabled=empty or len(history) < 12)
if empty:
    st.markdown("<p class='small c-orange'>Tell us what's in the meal first (or read the plate): a meal with no nutrition "
                "would just be a prediction without food.</p>", unsafe_allow_html=True)
elif len(history) < 12:
    st.markdown("<p class='small c-orange'>We need at least an hour of glucose before the meal.</p>", unsafe_allow_html=True)

meal = {"time": meal_time, "name": str(chosen.get("meal", "Your meal")), **vals, "doses": doses, "sessions": sessions}
sig = str(sorted((k, str(v)) for k, v in meal.items()))


def get_person():
    """Learn the person from everything before the meal (only when Predict! is pressed; kept for this session)."""
    with st.spinner(f"Getting to know {your_name or 'you'}: learning how you respond to food..."):
        return learn(history, info, f"meal-{found['history'].name}-{kind}-{a1c}-{history.index[-1]}-{len(history)}",
                     kind=str(profile["model"]) if "model" in profile else None,
                     exclude=str(profile["dataset_id"]) if "dataset_id" in profile else None)


if go_:
    person = get_person()
    with st.spinner("Drawing your next three hours..."):
        pred = predict_after_meal(person, history, meal)
    with st.spinner("Checking how well we know you (replaying your last meals)..."):
        st.session_state.meal_pred = (sig, pred, track_record(person, history), person["kind"])
    st.session_state.pop("meal_checked", None)

if "meal_pred" not in st.session_state:
    st.stop()
psig, pred, track, used = st.session_state.meal_pred
if psig != sig:
    st.markdown("<p class='c-orange'>You changed something. Hit Predict! again.</p>", unsafe_allow_html=True)
start_g = history["glucose_raw"].dropna()
st.markdown(f"<div class='rise'>{rise_text(pred, start_g.iloc[-1] if len(start_g) else pred['pred'].iloc[0], history['glucose'].median())}</div>"
            + commentary_html(meal_commentary(pred, history, meal))
            + accuracy_note(pred, track), unsafe_allow_html=True)
truth = st.session_state.get("meal_checked")
meal_plot(history, meal, pred, truth)

# ---------- Step 3 ----------
st.markdown("<div class='step-no'>Step 3</div><div class='step-title'>Tell us how we did</div>"
            "<p style='font-size:1.15rem'>Help improve our predictions so we can serve more people.</p>",
            unsafe_allow_html=True)
if len(later) >= 6:
    st.markdown(f"<p class='muted'>Your history file already has {len(later)} readings from after this meal. We hid them "
                "from the model; press Validate! to compare. Or drop another file below.</p>", unsafe_allow_html=True)
else:
    st.markdown("<p class='muted'>Drop your glucose for the hours after the meal: one file, several, a whole folder or a zip, "
                "as CSV, Excel or JSON. We use every reading after the meal. Trying a sample? Drag their later glucose onto "
                "the box.</p>", unsafe_allow_html=True)
sample_sets("meal_check", target="meal_check", key="samples_meal_check")
checks = drop_box("Your later glucose", DATA_TYPES, "meal_check") or ([found["truth"]] if found["truth"] is not None else [])
if st.button("Validate!", type="primary", disabled=not checks and len(later) < 6):
    readings, used = read_readings(checks, meal_time, pred.index[-1]) if checks else (None, [])
    st.session_state.meal_used = used
    readings = readings if checks else full[full.index >= meal_time]
    readings = readings if readings is not None else pd.DataFrame(columns=["glucose_raw"], index=pd.DatetimeIndex([]))
    got = readings["glucose_raw"][(readings.index >= meal_time) & (readings.index <= pred.index[-1])].dropna()
    st.session_state.meal_checked = got
    if len(got):
        learned = readings[readings.index >= meal_time]
        meal_row = pd.DataFrame({c: [meal[c]] for c in NUTR}, index=[meal_time.floor("5min")])
        learned = meal_row.combine_first(learned)
        prev = st.session_state.get("meal_learned")
        st.session_state.meal_learned = learned if prev is None else prev.combine_first(learned)
        contribute(history, learned, info, "meal")
    st.rerun()
if truth is not None:
    s = score(pred, truth)
    if s:
        k1, k2, _ = st.columns(3)
        k1.markdown(f"<div class='small'>This meal: typical miss (RMSE)</div><div class='big-number c-green'>{s['rmse']:.0f}</div>"
                    "<div class='small'>mg/dL from the predicted line</div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='small'>Truth inside the band</div><div class='big-number c-green'>{s['inside']:.0%}</div>"
                    f"<div class='small'>of {s['n']} readings</div>", unsafe_allow_html=True)
        if st.session_state.get("meal_used"):
            st.markdown("<p class='small'>Checked against " + ", ".join(st.session_state.meal_used) + ".</p>",
                        unsafe_allow_html=True)
        st.markdown("<p class='small'>Thank you! The green line in the plot above is what really happened. We've added this "
                    "meal and its readings to what we know about you, so your next prediction learns from it, and saved "
                    "them for the next time the model is retrained.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='c-orange'>We found no readings in the three hours after the meal.</p>", unsafe_allow_html=True)
