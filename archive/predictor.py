"""Predictor page: history, plan, predict, decomposition, accuracy."""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import sys
import json
import pickle
import warnings
from datetime import time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
for p_ in (ROOT, ROOT / "models"):
    if str(p_) not in sys.path:
        sys.path.insert(0, str(p_))
MODEL_FILES = ["clean", "common", "context", "events", "exogenous", "regressor", "overlay", "classifier", "decompose", "timeline"]


def code_version():
    """Changes whenever the weights or model code change."""
    files = [ROOT / "models" / "weights" / "models.pkl", ROOT / "timeline.py", *(ROOT / "models").glob("*.py")]
    return max(f.stat().st_mtime for f in files if f.exists())


@st.cache_resource
def fresh_modules(version):
    """Reload the model code once per change, before anything is imported from it."""
    import importlib
    for name in MODEL_FILES:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            importlib.import_module(name)
    return version


fresh_modules(code_version())
import decompose  # noqa: E402
from common import COLUMNS, WEIGHTS, MEAL_STEPS, from_csv, baseline, predict_window  # noqa: E402
from classifier import classify, load_classifier  # noqa: E402
from clean import remove_jitter  # noqa: E402
from context import WARP_FEATURES, meal_table  # noqa: E402
from events import exercise_table, quiet_table  # noqa: E402
from timeline import timeline  # noqa: E402

warnings.filterwarnings("ignore")
SHOWCASE, RESULTS = ROOT / "showcase", ROOT / "results"
FONT = "Libron, Georgia, serif"
NUTRIENTS = ["carbs", "protein", "fat", "fiber"]
KINDS = {"regressor": "Regressor", "overlay": "Overlay"}
MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snack", "Drink", "Other"]
EXERCISE_TYPES = ["Walk", "Run", "Cycle", "Strength", "Sport", "Housework", "Recorded", "Other"]
INTENSITY = ["light", "moderate", "hard"]
EXERCISE_LOAD = {"metabolic": {"light": 10, "moderate": 25, "hard": 45},   # activity added per 5 min (kcal)
                 "t1d": {"light": 3, "moderate": 5, "hard": 7}}          # (steps / 100)
LIMITS = {"carbs": 250.0, "protein": 150.0, "fat": 150.0, "fiber": 60.0, "calories": 4000.0, "insulin": 50.0}
MEAL_TEMPLATE = pd.DataFrame([
    {"time": "08:00", "meal": "Breakfast", "carbs": 50, "protein": 20, "fat": 15, "fiber": 5, "calories": 435, "insulin": 0},
    {"time": "13:00", "meal": "Lunch", "carbs": 60, "protein": 30, "fat": 20, "fiber": 6, "calories": 540, "insulin": 0},
    {"time": "19:30", "meal": "Dinner", "carbs": 70, "protein": 35, "fat": 25, "fiber": 8, "calories": 645, "insulin": 0}])
EXERCISE_TEMPLATE = pd.DataFrame([{"start": "17:30", "minutes": 30, "type": "Walk", "intensity": "moderate"}])
Z80 = 1.2816                                     # 80 % band = mean +- 1.28 sd
AFTER_MEAL = pd.Timedelta(hours=3)

# ---------- Flexoki colours ----------
LIGHT = dict(tx="#100F0F", tx2="#6F6E69", tx3="#B7B5AC", ui="#E6E4D9", bg="#FFFCF0", red="#AF3029", orange="#BC5215",
             yellow="#AD8301", green="#66800B", cyan="#24837B", blue="#205EA6", purple="#5E409D", magenta="#A02F6F")
DARK = dict(tx="#CECDC3", tx2="#878580", tx3="#575653", ui="#282726", bg="#100F0F", red="#D14D41", orange="#DA702C",
            yellow="#D0A215", green="#879A39", cyan="#3AA99F", blue="#4385BE", purple="#8B7EC8", magenta="#CE5D97")
P = DARK if getattr(st.context, "theme", None) is not None and st.context.theme.type == "dark" else LIGHT
ROLE = {"known": P["red"], "truth": P["green"], "regressor": P["magenta"], "overlay": P["cyan"], "meal": P["yellow"],
        "insulin": P["orange"], "exercise": P["purple"]}
TYPE_COLOR = {"healthy": "green", "prediabetes": "yellow", "t2d": "orange", "t1d": "magenta"}
SPIKE_COLORS = [P["red"], P["orange"], P["yellow"]]
COOL_COLORS = [P["blue"], P["cyan"], P["purple"]]
PART_COLORS = [P["orange"], P["cyan"], P["yellow"], P["purple"], P["green"], P["magenta"], P["red"], P["blue"]]

# text colours follow the page theme through CSS, so they stay right if the theme changes
css_vars = lambda pal: ";".join(f"--{k}:{v}" for k, v in pal.items())
st.markdown(f"""<style>
  :root {{{css_vars(LIGHT)}}}
  @media (prefers-color-scheme: dark) {{ :root {{{css_vars(DARK)}}} }}
  .block-container {{padding-top: 3.6rem;}}
  .who {{font-size: 2.2rem; line-height: 1.15; margin: 0 0 .5rem;}}
  .lbl {{color: var(--tx2); font-size: .82rem; letter-spacing: .04em; text-transform: uppercase; margin-right: .5rem;}}
  .facts {{display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: .25rem 2rem;
           font-size: 1.05rem; margin-bottom: .4rem;}}
  .facts b {{font-weight: 700;}}
  .stTabs [data-baseweb="tab-list"] {{gap: 2rem; margin-top: .6rem;}}
  .stTabs [data-baseweb="tab"] {{height: 3.2rem;}}
  .stTabs [data-baseweb="tab"] p {{font-size: 1.25rem;}}
  /* controls take the colour of what they set (sliders: the theme blue, hue-rotated) */
  [class*="st-key-c_carbs"] [data-testid="stSlider"] {{filter: hue-rotate(193deg);}}
  [class*="st-key-c_carbs"] [data-testid="stWidgetLabel"] p {{color: var(--yellow);}}
  [class*="st-key-c_protein"] [data-testid="stSlider"] {{filter: hue-rotate(152deg);}}
  [class*="st-key-c_protein"] [data-testid="stWidgetLabel"] p {{color: var(--red);}}
  [class*="st-key-c_fat"] [data-testid="stSlider"] {{filter: hue-rotate(170deg);}}
  [class*="st-key-c_fat"] [data-testid="stWidgetLabel"] p {{color: var(--orange);}}
  [class*="st-key-c_fiber"] [data-testid="stSlider"] {{filter: hue-rotate(221deg);}}
  [class*="st-key-c_fiber"] [data-testid="stWidgetLabel"] p {{color: var(--green);}}
  [class*="st-key-c_minutes"] [data-testid="stSlider"] {{filter: hue-rotate(48deg);}}
  [class*="st-key-c_minutes"] [data-testid="stWidgetLabel"] p {{color: var(--purple);}}
  [class*="st-key-c_intensity"] [data-testid="stSlider"] {{filter: hue-rotate(114deg);}}
  [class*="st-key-c_intensity"] [data-testid="stWidgetLabel"] p {{color: var(--magenta);}}
  [class*="st-key-c_portion"] [data-testid="stSlider"] {{filter: hue-rotate(323deg);}}
  [class*="st-key-c_portion"] [data-testid="stWidgetLabel"] p {{color: var(--cyan);}}
  [class*="st-key-c_opacity"] [data-testid="stSlider"] {{filter: hue-rotate(323deg);}}
  [class*="st-key-c_opacity"] [data-testid="stWidgetLabel"] p {{color: var(--cyan);}}
  [class*="st-key-c_smooth"] [data-testid="stSlider"] {{filter: hue-rotate(114deg);}}
  [class*="st-key-c_smooth"] [data-testid="stWidgetLabel"] p {{color: var(--magenta);}}
  [class*="st-key-c_cleaning"] [data-testid="stSlider"] {{filter: hue-rotate(170deg);}}
  [class*="st-key-c_cleaning"] [data-testid="stWidgetLabel"] p {{color: var(--orange);}}
  [class*="st-key-c_calories"] [data-testid="stWidgetLabel"] p, [class*="st-key-c_calories"] input {{color: var(--purple);}}
  [class*="st-key-c_insulin"] [data-testid="stWidgetLabel"] p {{color: var(--orange);}}
  .st-key-add_meal button, .st-key-remove_all button {{border-color: var(--yellow); color: var(--yellow);}}
  .st-key-add_ex button, .st-key-rec_ex button {{border-color: var(--purple); color: var(--purple);}}
  .muted {{color: var(--tx2); font-size: .9rem;}}
  .c-red {{color: var(--red);}} .c-orange {{color: var(--orange);}} .c-yellow {{color: var(--yellow);}}
  .c-green {{color: var(--green);}} .c-cyan {{color: var(--cyan);}} .c-blue {{color: var(--blue);}}
  .c-purple {{color: var(--purple);}} .c-magenta {{color: var(--magenta);}}
  .big {{font-size: 1.5rem;}}
  .step {{font-size: 1.05rem; margin: .4rem 0 .2rem;}}
</style>""", unsafe_allow_html=True)


def note(text):
    st.markdown(f"<span class='muted'>{text}</span>", unsafe_allow_html=True)


def rgba(color, alpha):
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def show(fig, height, **kw):
    layout = dict(height=height, font=dict(family=FONT, size=13), hovermode="x unified",
                  margin=dict(l=56, r=16, t=40, b=10), legend=dict(orientation="h", y=-0.18))
    layout.update(kw)
    fig.update_layout(**layout)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, width="stretch", theme="streamlit", config={"displayModeBar": False})


def line(fig, x, y, name, color, width=2.5, dash=None, unit="mg/dL", **kw):
    fig.add_scatter(x=x, y=y, name=name, line=dict(color=color, width=width, dash=dash), hovertemplate=f"%{{y:.0f}} {unit}", **kw)


def add_band(fig, x, mean, sd, color, name, **kw):
    if VIEW["bands"]:
        fig.add_scatter(x=x, y=mean + Z80 * sd, line=dict(width=0), showlegend=False, hoverinfo="skip", **kw)
        fig.add_scatter(x=x, y=mean - Z80 * sd, fill="tonexty", line=dict(width=0), fillcolor=rgba(color, VIEW["opacity"]),
                        name=name, hoverinfo="skip", **kw)


# ---------- Loading ----------

@st.cache_resource
def load_models(version):
    with open(WEIGHTS / "models.pkl", "rb") as f:
        return pickle.load(f), load_classifier()


@st.cache_resource
def photo_model():
    import photo2nutrition
    photo2nutrition.load()
    return photo2nutrition


@st.cache_data(show_spinner=False, max_entries=64)
def read_path(path, mtime):
    return from_csv(path)


def read(file):
    """Demo files are cached by path; uploads are read directly."""
    if file is None:
        return None
    if isinstance(file, Path):
        return read_path(str(file), file.stat().st_mtime)
    return from_csv(file)


def by_time_of_day(history, col):
    return history[col].groupby(history.index.time).median()


def meal_name(t):
    h = t.hour + t.minute / 60
    return "Snack" if h < 4 or h >= 21.5 else "Breakfast" if h < 11 else "Lunch" if h < 16 else "Dinner"


def as_time(v):
    if isinstance(v, dtime):
        return v
    return pd.Timestamp(str(v)).time() if len(str(v)) > 5 else pd.Timestamp(f"2000-01-01 {v}").time()


def smooth(y):
    """Display smoothing only (models are not affected)."""
    if y is None or VIEW["method"] == "off" or y.notna().sum() < 3:
        return y
    if VIEW["method"] == "jitter removal":
        return remove_jitter(y, VIEW["rel"] / 100)
    return y.rolling(f"{VIEW['window']}min", center=True, min_periods=1).mean()


def runs(index, mask):
    m = np.r_[False, mask, False]
    edges = np.flatnonzero(np.diff(m.astype(int)))
    return [(index[a], index[min(b, len(index) - 1)]) for a, b in zip(edges[::2], edges[1::2])]


def meal_spikes(series):
    meals = meal_table(series)
    if meals.empty:
        return meals
    g = series["glucose"].to_numpy()
    rise, peak = [], []
    for a in meals["step"]:
        w = g[a: a + 25]
        ok = len(w) == 25 and np.isfinite(w).all()
        rise.append(w.max() - w[0] if ok else np.nan)
        peak.append(int(np.argmax(w)) * 5 if ok else np.nan)
    meals["time"] = series.index[meals["step"]]
    meals["carbs"] = series["carbs"].to_numpy()[meals["step"]]
    meals["calories"] = series["calories"].to_numpy()[meals["step"]]
    meals["rise"], meals["peak_min"] = rise, peak
    meals["day"] = (meals["time"] - pd.Timedelta(hours=4)).dt.date
    return meals


def daily_summary(history):
    meals = meal_spikes(history)
    g = history["glucose_raw"]
    days = pd.DataFrame({"mean": g.groupby(g.index.date).mean(), "max": g.groupby(g.index.date).max(),
                         "% > 140": (g > 140).groupby(g.index.date).mean() * 100})
    if not meals.empty:
        per = meals.groupby("day").agg(meals=("slot", "size"), carbs=("carbs", "sum"), kcal=("calories", "sum"))
        top = meals.dropna(subset=["rise"]).sort_values("rise").groupby("day").tail(1).set_index("day")
        per["largest spike"] = top["rise"]
        per["at meal"] = top["slot"]
        days = days.join(per, how="left")
    return days.round(1)


def standard_day(model, post, history, base):
    """Model response to the same 50 g carb meal as meal 1 / 2 / 3 of a usual day."""
    start = history.index[-1].normalize() + pd.Timedelta(days=1)
    grid = pd.date_range(history.index[0], start + pd.Timedelta(days=1), freq="5min")
    s = history.reindex(grid)
    new = s.index > history.index[-1]
    for c in ["activity", "hr", "basal"]:
        typical = by_time_of_day(history, c)
        s.loc[new, c] = [typical.get(t, np.nan) for t in s.index[new].time]
    for c in ["carbs", "protein", "fat", "fiber", "calories", "bolus", "basal", "activity"]:
        s[c] = s[c].fillna(0)
    out = {}
    for slot, hour in [(1, 8), (2, 13), (3, 19)]:
        t = start + pd.Timedelta(hours=hour)
        s.loc[t, ["carbs", "protein", "fat", "fiber", "calories"]] = [50, 20, 15, 5, 435]
        a = s.index.get_loc(t)
        s.iloc[a - 6: a + 1, s.columns.get_loc("glucose")] = base
        mean, sd = model.predict_with_sd(s, a, post, base)
        i = int(np.argmax(mean[0][:24]))
        out[f"{hour}:00, meal {slot}"] = {"spike": f"{mean[0][i] - base:.0f} ± {sd[0][i]:.0f}", "peak (min)": (i + 1) * 5}
    return pd.DataFrame(out).T


# ---------- Sidebar: patient and display ----------
version = code_version()
with st.spinner("Loading the models..."):
    bundle, clf = load_models(version)
demo_patients = sorted(p.name for p in SHOWCASE.iterdir() if p.is_dir()) if SHOWCASE.exists() else []
valid_files = {p.parent.name: p for p in sorted(SHOWCASE.glob("*/validation.csv"))}

with st.sidebar:
    st.markdown("#### Patient")
    source = st.radio("Source", ["Demo patients", "Upload a history"], label_visibility="collapsed")
    info, history, day_inputs, truth, folder = {}, None, None, None, None
    if source == "Demo patients" and demo_patients:
        pick = st.selectbox("Patient", demo_patients, label_visibility="collapsed")
        folder = SHOWCASE / pick
        history = read(folder / "history.csv")
        day_inputs = read(folder / "day8_inputs.csv") if (folder / "day8_inputs.csv").exists() else None
        truth = json.loads((folder / "info.json").read_text())
        if st.checkbox("Tell the model their diabetes type", value=False):
            info["group"] = truth["group"]
        patient_key = pick
    else:
        hist_file = st.file_uploader("History CSV", type="csv", help="Columns: time, then any of " + ", ".join(COLUMNS))
        history = read(hist_file)
        kind = st.selectbox("Diabetes type", ["Auto", "healthy", "prediabetes", "t2d", "t1d"])
        if kind != "Auto":
            info["group"] = kind
        a1c = st.number_input("Lab A1c % (0 = unknown)", 0.0, 15.0, 0.0, 0.1)
        if a1c > 0:
            info["a1c"] = a1c
        day_inputs = read(st.file_uploader("Recorded inputs for the planned day (optional)", type="csv"))
        patient_key = getattr(hist_file, "name", "none")
    opts = ["none"] + list(valid_files)
    v_choice = st.selectbox("Validation file (the truth)", opts, key=f"valid_{patient_key}",
                            index=opts.index(patient_key) if patient_key in opts else 0,
                            help="Readings before the prediction window are given to the model. Readings inside it are "
                                 "only used to score, and stay hidden until you press Show truth.")
    v_upload = st.file_uploader("Or upload one (time, glucose)", type="csv", key="valid_upload")

    st.markdown("#### Display")
    VIEW = {"method": st.selectbox("Smoothing", ["off", "jitter removal", "moving average"], index=1)}
    if VIEW["method"] == "jitter removal":
        with st.container(key="c_smooth_a"):
            VIEW["rel"] = st.slider("Ignore wiggles below (% of level)", 0.5, 10.0, 3.0, 0.5, key="view_rel")
    elif VIEW["method"] == "moving average":
        with st.container(key="c_smooth_b"):
            VIEW["window"] = st.slider("Window (min)", 10, 90, 30, 5, key="view_window")
    VIEW["bands"] = st.checkbox("Show 80 % uncertainty bands", value=True)
    with st.container(key="c_opacity"):
        VIEW["opacity"] = st.slider("Band opacity", 0.05, 0.5, 0.18, 0.05, key="view_opacity")
    with st.container(key="c_cleaning"):
        clean_rel = st.slider("Model input cleaning (% of level)", 0.0, 5.0, 1.0, 0.5, key="clean_rel",
                              help="Jitter removed from the history before the model learns from it. Training used 1 %.")

if history is None:
    st.info("Select a demo patient or upload a history CSV.")
    st.stop()

history = history.assign(glucose=remove_jitter(history["glucose_raw"], clean_rel / 100) if clean_rel > 0 else history["glucose_raw"])
route = classify(info, history, clf)
backend, group = route["backend"], route["group"]
base = baseline(history)
models = {k: bundle[(backend, k)] for k in KINDS}
posts = st.session_state.setdefault("posts", {})
for k, m in models.items():
    key = (patient_key, backend, group, k, clean_rel, version)
    if key not in posts:
        with st.spinner(f"Learning this patient ({KINDS[k].lower()})..."):
            posts[key] = m.posterior(history, group if group in m.groups else next(iter(m.groups)), base)
post = {k: posts[(patient_key, backend, group, k, clean_rel, version)] for k in KINDS}

# ---------- Header ----------
days = (history.index[-1] - history.index[0]).total_seconds() / 86400
group_name = {"healthy": "healthy", "prediabetes": "prediabetic", "t2d": "type 2", "t1d": "type 1"}[group]
recorded = f" <span class='muted'>(recorded: {truth['group']}, {truth['source']})</span>" if truth else ""
probs = (" " + ", ".join(f"{g} {p:.0%}" for g, p in route["probabilities"].items())) if route["probabilities"] else ""
st.markdown(f"""<div class='who'><span class='lbl'>Patient</span>{patient_key}</div>
<div class='facts'>
  <div><span class='lbl'>Diabetes type</span><b class='c-{TYPE_COLOR[group]}'>{group_name}</b>{recorded}</div>
  <div><span class='lbl'>Model</span>{'type 1, with insulin doses' if backend == 't1d' else 'metabolic (makes own insulin)'}</div>
  <div><span class='lbl'>History</span>{days:.1f} days, {int((history['carbs'] > 0).sum())} meals logged</div>
  <div><span class='lbl'>Usual glucose</span>{base:.0f} mg/dL</div>
</div>
<div class='muted'>Type decided by {route['reason']}.{probs}</div>""", unsafe_allow_html=True)

# ---------- Plan state ----------
day0 = (history.index[-1] + pd.Timedelta(minutes=5)).normalize()
if st.session_state.get("plan_for") != patient_key:
    st.session_state.plan_for = patient_key
    for k in [k for k in st.session_state if k.startswith(("meal_", "ex_", "doses", "show_truth", "truth_for", "pred_", "dec_", "tl_"))]:
        del st.session_state[k]
    meals, doses = [], pd.DataFrame({"time": pd.Series(dtype="datetime64[ns]"), "units": pd.Series(dtype=float)})
    if day_inputs is not None:
        bolus = day_inputs[day_inputs["bolus"] > 0]["bolus"]
        used = set()
        for t, r in day_inputs[day_inputs["carbs"] > 0].iterrows():
            near = bolus[(bolus.index >= t - pd.Timedelta(minutes=30)) & (bolus.index <= t + pd.Timedelta(minutes=30))]
            used |= set(near.index)
            meals.append({"meal": meal_name(t), "date": t.date(), "time": t.time(), **{n: float(r[n]) for n in NUTRIENTS},
                          "calories": float(r["calories"]) or 4 * r["carbs"] + 4 * r["protein"] + 9 * r["fat"], "insulin": float(near.sum())})
        rest = bolus[~bolus.index.isin(list(used))]
        doses = pd.DataFrame({"time": rest.index, "units": rest.round(2).to_numpy()})
    if not meals:
        meals = [{"meal": r["meal"], "date": day0.date(), "time": as_time(r["time"]), **{n: float(r[n]) for n in NUTRIENTS},
                  "calories": float(r["calories"]), "insulin": 0.0} for _, r in MEAL_TEMPLATE.iterrows()]
    st.session_state.meal_ids = list(range(len(meals)))
    for i, m in enumerate(meals):
        for f, v in m.items():
            st.session_state[f"meal_{i}_{f}"] = min(v, LIMITS[f]) if f in LIMITS else v
    st.session_state.ex_ids = []
    if day_inputs is not None:                      # the demo day's exercise, detected from its recorded activity
        ex = exercise_table(pd.concat([history, day_inputs.drop(columns=["glucose", "glucose_raw"], errors="ignore")]).sort_index())
        both = pd.concat([history, day_inputs]).sort_index().index
        ex = ex[ex["step"] >= len(history)]
        for i, (_, r) in enumerate(ex.iterrows()):
            t0 = both[int(r["step"])]
            st.session_state.ex_ids.append(i)
            st.session_state.update({f"ex_{i}_type": "Recorded", f"ex_{i}_date": t0.date(), f"ex_{i}_start": t0.time(),
                                     f"ex_{i}_minutes": int(min(max(r["minutes"], 5), 180)),
                                     f"ex_{i}_intensity": "hard" if r["intensity"] > 5 else "moderate" if r["intensity"] > 3 else "light"})
    st.session_state.doses = doses
    st.session_state.window = (day0 + pd.Timedelta(hours=5), day0 + pd.Timedelta(hours=23))
    st.session_state.pred_mode = "After a meal"

# apply changes requested by a click before the widgets are drawn
for k_ in ("pred_mode", "pred_meal", "dec_meal"):
    if f"_set_{k_}" in st.session_state:
        st.session_state[k_] = st.session_state.pop(f"_set_{k_}")

# lazy tabs: only the open tab's heavy content runs (Plan always runs: its widgets hold the meals)
tab_hist, tab_plan, tab_pred, tab_dec, tab_acc = st.tabs(["History", "Plan", "Predict", "Decomposition", "Accuracy"],
                                                         default="Predict", key="main_tab", on_change="rerun")

# ---------- Plan (first: everything else needs the meals) ----------
with tab_plan:
    b1, b2, b3, b4 = st.columns([2, 1, 1, 1])
    meal_csv = b1.file_uploader("Load meals from a CSV", type="csv", key="meal_csv")
    b2.download_button("CSV template", MEAL_TEMPLATE.to_csv(index=False), "meals_template.csv", width="stretch")
    if b3.button("Add a meal", width="stretch", key="add_meal"):
        new = max(st.session_state.meal_ids, default=-1) + 1
        st.session_state.meal_ids.append(new)
        st.session_state.update({f"meal_{new}_meal": "Snack", f"meal_{new}_date": day0.date(), f"meal_{new}_time": dtime(15, 30),
                                 f"meal_{new}_carbs": 25.0, f"meal_{new}_protein": 5.0, f"meal_{new}_fat": 5.0,
                                 f"meal_{new}_fiber": 2.0, f"meal_{new}_calories": 165.0, f"meal_{new}_insulin": 0.0})
    if b4.button("Remove all", width="stretch", key="remove_all"):
        st.session_state.meal_ids = []
    if meal_csv is not None and st.session_state.get("meal_csv_done") != meal_csv.name:
        df = pd.read_csv(meal_csv)
        df.columns = df.columns.str.strip().str.lower()
        st.session_state.meal_ids = list(range(len(df)))
        for i, r in df.iterrows():
            t = as_time(r["time"])
            d = pd.Timestamp(r["date"]).date() if "date" in df else day0.date()
            vals = {n: min(float(r.get(n, 0) or 0), LIMITS[n]) for n in NUTRIENTS}
            st.session_state.update({f"meal_{i}_meal": str(r.get("meal", meal_name(pd.Timestamp.combine(d, t)))), f"meal_{i}_date": d,
                                     f"meal_{i}_time": t, **{f"meal_{i}_{n}": v for n, v in vals.items()},
                                     f"meal_{i}_calories": min(float(r.get("calories", 0) or 4 * vals["carbs"] + 4 * vals["protein"] + 9 * vals["fat"]), LIMITS["calories"]),
                                     f"meal_{i}_insulin": float(r.get("insulin", 0) or 0)})
        st.session_state.meal_csv_done = meal_csv.name
        st.rerun()

    meals_plan = []
    order = sorted(st.session_state.meal_ids, key=lambda i: (st.session_state.get(f"meal_{i}_date", day0.date()),
                                                             st.session_state.get(f"meal_{i}_time", dtime(12))))
    for n, i in enumerate(order):
        with st.container(border=True):
            c1_, c2_, c3_, c4_, c5_ = st.columns([1.3, 1, 1, 1, 0.35])
            if st.session_state.get(f"meal_{i}_meal") not in MEAL_TYPES:
                st.session_state[f"meal_{i}_meal"] = "Other"
            st.session_state.setdefault(f"meal_{i}_date", day0.date())
            kind_ = c1_.selectbox(f"Meal {n + 1}", MEAL_TYPES, key=f"meal_{i}_meal")
            d = c2_.date_input("Date", key=f"meal_{i}_date")
            t = c3_.time_input("Time", key=f"meal_{i}_time", step=timedelta(minutes=5))
            with c4_.container(key=f"c_insulin_{i}"):
                insulin = st.number_input("Insulin (U)", 0.0, 50.0, step=0.5, key=f"meal_{i}_insulin", disabled=backend != "t1d")
            with c5_.popover("", icon=":material/more_vert:"):
                if st.button("Remove this meal", key=f"meal_{i}_del", width="stretch"):
                    st.session_state.meal_ids.remove(i)
                    st.rerun()
                photo = st.file_uploader("Estimate from a photo", type=["jpg", "jpeg", "png"], key=f"meal_{i}_photo")
                with st.container(key=f"c_portion_{i}"):
                    portion = st.select_slider("Portion", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0], value=1.0, key=f"meal_{i}_portion")
                if photo is not None and st.button("Estimate", key=f"meal_{i}_go", width="stretch"):
                    from PIL import Image
                    res = photo_model().estimate(Image.open(photo), portion)
                    for k_, v in res["nutrition"].items():
                        st.session_state[f"meal_{i}_{k_}"] = round(min(float(v), LIMITS[k_]), 1)
                    st.session_state[f"meal_{i}_note"] = f"{res['source']}: {res['detail']}"
                    st.rerun()
            cols5 = st.columns(5)
            vals = {}
            for col_, (n_, label_, top_) in zip(cols5, [("carbs", "Carbs (g)", 250.0), ("protein", "Protein (g)", 150.0),
                                                        ("fat", "Fat (g)", 150.0), ("fiber", "Fiber (g)", 60.0)]):
                with col_.container(key=f"c_{n_}_{i}"):
                    vals[n_] = st.slider(label_, 0.0, top_, step=1.0, key=f"meal_{i}_{n_}")
            est = 4 * vals["carbs"] + 4 * vals["protein"] + 9 * vals["fat"]
            with cols5[4].container(key=f"c_calories_{i}"):
                vals["calories"] = st.number_input("Calories", 0.0, 4000.0, step=10.0, key=f"meal_{i}_calories", help=f"From macros: {est:.0f}")
            if f"meal_{i}_note" in st.session_state:
                note(st.session_state[f"meal_{i}_note"])
            meals_plan.append({"meal": kind_, "datetime": pd.Timestamp.combine(d, t), **vals, "insulin": insulin})

    if backend == "t1d":
        st.markdown("##### Other insulin doses")
        doses = st.data_editor(st.session_state.doses, num_rows="dynamic", key="doses_editor", width="stretch",
                               column_config={"time": st.column_config.DatetimeColumn("time", format="YYYY-MM-DD HH:mm", step=300),
                                              "units": st.column_config.NumberColumn("units", min_value=0.0, max_value=50.0, step=0.1)})
    else:
        doses = st.session_state.doses

    st.markdown("##### Exercise")
    recorded_act = day_inputs is not None and day_inputs["activity"].abs().sum() > 0
    use_recorded = st.checkbox("Use the recorded activity and heart rate", value=recorded_act, disabled=not recorded_act,
                               help="Unticked: the person's usual activity by time of day, plus the sessions below.")
    e1, e2, e3, e4 = st.columns([2, 1, 1, 1])
    ex_csv = e1.file_uploader("Load exercise from a CSV", type="csv", key="ex_csv")
    e2.download_button("CSV template", EXERCISE_TEMPLATE.to_csv(index=False), "exercise_template.csv", width="stretch", key="ex_tpl")
    if e3.button("Add a session", width="stretch", key="add_ex"):
        new = max(st.session_state.ex_ids, default=-1) + 1
        st.session_state.ex_ids.append(new)
        st.session_state.update({f"ex_{new}_type": "Walk", f"ex_{new}_date": day0.date(), f"ex_{new}_start": dtime(17, 30),
                                 f"ex_{new}_minutes": 30, f"ex_{new}_intensity": "moderate"})
    demo_ex = folder / "day8_exercise.csv" if folder is not None else None
    if demo_ex is not None and demo_ex.exists() and e4.button("Recorded sessions", width="stretch", key="rec_ex"):
        ex_csv, st.session_state.ex_csv_done = demo_ex, None
    if ex_csv is not None and st.session_state.get("ex_csv_done") != getattr(ex_csv, "name", str(ex_csv)):
        df = pd.read_csv(ex_csv)
        df.columns = df.columns.str.strip().str.lower()
        st.session_state.ex_ids = list(range(len(df)))
        for i, r in df.iterrows():
            st.session_state.update({f"ex_{i}_type": str(r.get("type", "Other")), f"ex_{i}_date": day0.date(),
                                     f"ex_{i}_start": as_time(r["start"]), f"ex_{i}_minutes": int(min(max(r.get("minutes", 30), 5), 180)),
                                     f"ex_{i}_intensity": str(r.get("intensity", "moderate"))})
        st.session_state.ex_csv_done = getattr(ex_csv, "name", str(ex_csv))
        st.rerun()
    sessions = []
    for i in list(st.session_state.ex_ids):
        with st.container(border=True):
            x1, x2, x3, x4, x5, x6 = st.columns([1.1, 1, 1, 1.5, 1.4, 0.35])
            if st.session_state.get(f"ex_{i}_type") not in EXERCISE_TYPES:
                st.session_state[f"ex_{i}_type"] = "Other"
            st.session_state.setdefault(f"ex_{i}_date", day0.date())
            ex_kind = x1.selectbox("Activity", EXERCISE_TYPES, key=f"ex_{i}_type")
            d = x2.date_input("Date", key=f"ex_{i}_date")
            t = x3.time_input("Start", key=f"ex_{i}_start", step=timedelta(minutes=5))
            with x4.container(key=f"c_minutes_{i}"):
                mins = st.slider("Minutes", 5, 180, step=5, key=f"ex_{i}_minutes")
            with x5.container(key=f"c_intensity_{i}"):
                inten = st.select_slider("Intensity", INTENSITY, key=f"ex_{i}_intensity")
            with x6.popover("", icon=":material/more_vert:"):
                if st.button("Remove", key=f"ex_{i}_del", width="stretch"):
                    st.session_state.ex_ids.remove(i)
                    st.rerun()
            sessions.append((pd.Timestamp.combine(d, t), mins, inten, ex_kind))

meal_list = [m for m in sorted(meals_plan, key=lambda m: m["datetime"]) if m["carbs"] > 0]
SIG = str((patient_key, clean_rel, version, [(m["datetime"], m["carbs"], m["protein"], m["fat"], m["fiber"], m["calories"],
                                              m["insulin"]) for m in meals_plan], sessions,
           pd.DataFrame(doses).dropna().values.tolist(), use_recorded, backend, group, v_choice, getattr(v_upload, "name", None)))
meal_label = lambda m: f"{m['meal']} {m['datetime']:%a %H:%M}"

# ---------- Validation (glucose for the predicted period) ----------
validation = read(v_upload) if v_upload is not None else (read(valid_files[v_choice]) if v_choice in valid_files else None)
all_glucose = history["glucose_raw"] if validation is None else history["glucose_raw"].combine_first(validation["glucose_raw"])
lo = history.index[0].normalize() - pd.Timedelta(days=1)
last = max(all_glucose.index[-1], max([m["datetime"] for m in meal_list], default=all_glucose.index[-1]))
hi = last.normalize() + pd.Timedelta(days=3)                                     # +2 days after the last day with data
glucose_runs = runs(all_glucose.index, all_glucose.notna().to_numpy())
TL_COLORS = {"before": P["red"], "inside": P["green"], "after": P["blue"], "none": P["tx3"], "band": P["tx3"],
             "meal": P["yellow"], "handle": P["blue"], "text": P["tx"], "muted": P["tx2"], "bg": P["bg"]}
meal_dots = [(m["datetime"], f"{meal_label(m)}: {m['carbs']:.0f} g carbs, {m['calories']:.0f} kcal") for m in meal_list]


def build_series(start):
    """All inputs on one 5-min grid. Glucose is everything known before `start` (history + validation)."""
    end_ = st.session_state.window[1]
    plan_end = max([end_] + [m["datetime"] for m in meals_plan] + [s_[0] for s_ in sessions]) + pd.Timedelta(hours=5)
    grid = pd.date_range(history.index[0], max(plan_end, history.index[-1] + pd.Timedelta(hours=5)), freq="5min")
    s = history.reindex(grid)
    future = s.index > history.index[-1]
    for c in ["carbs", "protein", "fat", "fiber", "calories", "bolus", "basal", "activity"]:
        s[c] = s[c].fillna(0)
    for c in ["activity", "hr", "basal"]:
        typical = by_time_of_day(history, c)
        s.loc[future, c] = [typical.get(t, np.nan) for t in s.index[future].time]
    if use_recorded:
        for c in ["activity", "hr", "basal"]:
            given = day_inputs[c].reindex(grid)
            s.loc[future & given.notna().to_numpy(), c] = given
    s.loc[future, ["carbs", "protein", "fat", "fiber", "calories", "bolus"]] = 0.0
    s[["activity", "basal"]] = s[["activity", "basal"]].fillna(0)
    for m in meals_plan:
        t = m["datetime"].floor("5min")
        if t in s.index and m["carbs"] + m["protein"] + m["fat"] > 0:
            for n in NUTRIENTS + ["calories"]:
                s.loc[t, n] += m[n]
            if backend == "t1d":
                s.loc[t, "bolus"] += m["insulin"]
    if backend == "t1d":
        for _, d in pd.DataFrame(doses).dropna().iterrows():
            t = pd.Timestamp(d["time"]).floor("5min")
            if t in s.index:
                s.loc[t, "bolus"] += float(d["units"])
    for t0, mins, inten, kind_ in sessions:
        if use_recorded and kind_ == "Recorded":
            continue                                # already in the recorded activity; the card is for show
        span = pd.date_range(t0.floor("5min"), periods=max(mins // 5, 1), freq="5min")
        s.loc[s.index.isin(span), "activity"] += EXERCISE_LOAD[backend][inten]
    if validation is not None:
        s["glucose"] = s["glucose"].fillna(validation["glucose"].reindex(grid).where(grid < start))
    s.loc[s.index >= start, "glucose"] = np.nan
    return s


def mark_events(fig, t_start, t_end):
    for m in meal_list:
        t = m["datetime"]
        if t_start <= t <= t_end:
            fig.add_vline(x=t, line=dict(color=P["yellow"], dash="dot", width=1.2))
            fig.add_annotation(x=t, y=1.0, yref="paper", yanchor="bottom", showarrow=False, font=dict(size=12, color=P["yellow"]),
                               text=f"{m['meal']} · {m['carbs']:.0f} g · {m['calories']:.0f} kcal")
    for t0, mins, inten, _ in sessions:
        if t0 <= t_end and t0 + pd.Timedelta(minutes=mins) >= t_start:
            fig.add_vrect(x0=t0, x1=t0 + pd.Timedelta(minutes=mins), fillcolor=P["purple"], opacity=0.12, line_width=0)


# ---------- History ----------
with tab_hist:
    if tab_hist.open:
        meals_hist = meal_spikes(history)
        o1, o2, o3, o4 = st.columns(4)
        show_raw = o1.checkbox("Raw CGM", value=True)
        show_clean = o2.checkbox("Cleaned (model input)", value=False)
        show_smooth = o3.checkbox("Smoothed (display)", value=VIEW["method"] != "off")
        show_events = o4.checkbox("Exercise and quiet blocks", value=True)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.56, 0.2, 0.24], vertical_spacing=0.06,
                            subplot_titles=["Glucose (mg/dL)", "Carbs (g) and insulin", "Activity and heart rate"],
                            specs=[[{}], [{}], [{"secondary_y": True}]])
        if show_raw:
            line(fig, history.index, history["glucose_raw"], "raw", P["red"], 1.1, row=1, col=1)
        if show_clean:
            line(fig, history.index, history["glucose"], f"cleaned ({clean_rel:g} %)", P["blue"], 1.4, row=1, col=1)
        if show_smooth and VIEW["method"] != "off":
            line(fig, history.index, smooth(history["glucose_raw"]), VIEW["method"], P["magenta"], 2, row=1, col=1)
        if not meals_hist.empty:
            fig.add_bar(x=meals_hist["time"], y=meals_hist["carbs"], name="carbs", marker_color=P["yellow"], width=20 * 60 * 1000,
                        customdata=np.c_[meals_hist["slot"], meals_hist["calories"], meals_hist["rise"]],
                        hovertemplate="meal %{customdata[0]}: %{y:.0f} g, %{customdata[1]:.0f} kcal, spike %{customdata[2]:.0f}<extra></extra>",
                        row=2, col=1)
        if history["bolus"].sum() > 0:
            b = history[history["bolus"] > 0]
            fig.add_scatter(x=b.index, y=b["bolus"] * 10, mode="markers", name="insulin (U x10)", hovertemplate="%{y:.1f}",
                            marker=dict(color=P["orange"], symbol="triangle-up", size=7), row=2, col=1)
        line(fig, history.index, history["activity"], "activity", P["purple"], 1, unit="", row=3, col=1)
        if history["hr"].notna().any():
            fig.add_scatter(x=history.index, y=history["hr"], name="heart rate", line=dict(color=P["green"], width=1),
                            hovertemplate="%{y:.0f} bpm", row=3, col=1, secondary_y=True)
        if show_events:
            ex_hist = exercise_table(history)
            for _, r in ex_hist.iterrows():
                t0 = history.index[int(r["step"])]
                fig.add_vrect(x0=t0, x1=t0 + pd.Timedelta(minutes=int(r["minutes"])), fillcolor=P["purple"], opacity=0.2, line_width=0, row=1, col=1)
            for _, r in quiet_table(history, ex_hist).iterrows():
                t0 = history.index[int(r["step"])]
                fig.add_vrect(x0=t0, x1=t0 + pd.Timedelta(hours=4), fillcolor=P["tx3"], opacity=0.12, line_width=0, row=1, col=1)
        show(fig, 640)
        changed = (history["glucose"] - history["glucose_raw"]).abs()
        note(f"Cleaning changed {(changed > 0.05).mean():.1%} of readings (mean {changed[changed > 0.05].mean() if (changed > 0.05).any() else 0:.1f} mg/dL). "
             "Shaded: detected exercise (purple) and 4-hour blocks with no meal or exercise (grey).")
        left, right = st.columns([3, 2])
        left.markdown("##### Days")
        left.dataframe(daily_summary(history), width="stretch")
        with right:
            st.markdown("##### By meal of the day")
            if not meals_hist.empty and meals_hist["rise"].notna().any():
                slots = meals_hist.groupby("slot").agg(n=("rise", "size"), carbs=("carbs", "mean"), spike=("rise", "mean"), peak_min=("peak_min", "mean"))
                slots["spike / 10 g"] = meals_hist.groupby("slot").apply(lambda d: (d["rise"] / d["carbs"].clip(lower=5) * 10).mean())
                slots.index = [f"meal {i}" for i in slots.index]
                st.dataframe(slots.round(1), width="stretch")
            st.markdown("##### The same 50 g meal, as the models see this person (± 1 sd)")
            st.dataframe(pd.concat({KINDS[k]: standard_day(models[k], post[k], history, base) for k in KINDS}, axis=1), width="stretch")

# ---------- Predict ----------
def go_meal(step_):
    """Previous / next meal buttons: move to the neighbouring meal, in after-a-meal mode."""
    labels_ = [meal_label(m_) for m_ in meal_list]
    i_ = labels_.index(st.session_state.pred_meal) if st.session_state.get("pred_meal") in labels_ else 0
    st.session_state.pred_meal = labels_[min(max(i_ + step_, 0), len(labels_) - 1)]
    st.session_state.pred_mode = "After a meal"


def go_day():
    """Whole day button: a window from the day's first meal to 3 h after its last meal."""
    labels_ = [meal_label(m_) for m_ in meal_list]
    cur = meal_list[labels_.index(st.session_state.pred_meal)] if st.session_state.get("pred_meal") in labels_ else meal_list[0]
    day_meals = [m_["datetime"] for m_ in meal_list if m_["datetime"].date() == cur["datetime"].date()]
    st.session_state.window = (min(day_meals) - pd.Timedelta(hours=1), max(day_meals) + AFTER_MEAL)
    st.session_state.pred_mode = "A time window"


@st.cache_data(show_spinner=False, max_entries=128)
def cached_forecast(_model, _series, _post, key, start_, end_, base_):
    p_, sd_ = predict_window(_model, _series, _post, start_, end_, base_)
    return pd.DataFrame({"pred": p_, "sd": sd_})


def toggle_truth(window_):
    st.session_state.show_truth = not st.session_state.get("show_truth", False)
    st.session_state.truth_for = window_


@st.fragment
def predict_panel():
    """Its own fragment: showing the truth or picking models only redraws this part."""
    labels = [meal_label(m) for m in meal_list]
    st.session_state.setdefault("pred_mode", "After a meal")
    if labels and st.session_state.get("pred_meal") not in labels:      # default: the first real meal (from 5 am)
        daytime = [meal_label(m) for m in meal_list if m["datetime"].hour >= 5]
        st.session_state.pred_meal = (daytime or labels)[0]
    n1, n2, n3, n4, n5 = st.columns([1.6, 1.6, 0.8, 0.8, 0.8], vertical_alignment="bottom")
    mode = n1.radio("Predict", ["After a meal", "A time window"], key="pred_mode", horizontal=True)
    if labels:
        n2.selectbox("Meal", labels, key="pred_meal", disabled=mode != "After a meal")
        idx_now = labels.index(st.session_state.pred_meal)
        nav = [n3.button("Previous meal", width="stretch", on_click=go_meal, args=(-1,), disabled=idx_now == 0),
               n4.button("Next meal", width="stretch", on_click=go_meal, args=(1,), disabled=idx_now == len(labels) - 1),
               n5.button("Whole day", width="stretch", on_click=go_day)]
        if any(nav):
            st.rerun()                                       # the window changed: refresh every tab
        if mode == "After a meal":
            chosen = meal_list[idx_now]
            st.session_state.window = (chosen["datetime"], chosen["datetime"] + AFTER_MEAL)
    elif mode == "After a meal":
        n2.warning("Add a meal on the Plan tab first.")
    start, end = st.session_state.window
    moved, clicked = timeline("tl_pred", lo, hi, (start, end), glucose_runs, meal_dots, TL_COLORS,
                              selected=start if mode == "After a meal" else None)
    if moved is not None:
        st.session_state.window = moved
        st.session_state["_set_pred_mode"] = "A time window"
        st.rerun()
    if clicked is not None:
        hit = [m for m in meal_list if abs((m["datetime"] - clicked).total_seconds()) < 60]
        if hit:
            st.session_state["_set_pred_mode"] = "After a meal"
            st.session_state["_set_pred_meal"] = meal_label(hit[0])
            st.rerun()

    series = build_series(start)
    known_until = series["glucose"].last_valid_index()
    preds = {k: cached_forecast(models[k], series[COLUMNS], post[k], (SIG, k), start, end, base) for k in KINDS}

    # truth: hidden until asked for, and hidden again whenever the window changes
    if st.session_state.get("truth_for") != (start, end):
        st.session_state.show_truth = False
    has_truth = validation is not None and \
        validation["glucose_raw"][(validation.index >= start) & (validation.index <= end)].notna().any()
    t1_, t2_, t3_ = st.columns([1, 1.6, 3], vertical_alignment="center")
    t1_.button("Hide truth" if st.session_state.get("show_truth") else "Show truth", key="truth_btn", width="stretch",
               disabled=not has_truth, type="primary", on_click=toggle_truth, args=((start, end),))
    kinds = t2_.pills("Models", list(KINDS), default=list(KINDS), selection_mode="multi", format_func=KINDS.get,
                      key="pred_kinds", label_visibility="collapsed") or []
    if not has_truth:
        t3_.markdown("<span class='muted'>No validation glucose in this window. Pick a validation file in the sidebar.</span>",
                     unsafe_allow_html=True)
    if v_choice not in ("none", patient_key) and v_upload is None:
        st.warning(f"The validation file belongs to {v_choice}.")
    note(f"{'After ' + st.session_state.pred_meal if mode == 'After a meal' and labels else 'Window'}: "
         f"{start:%a %d %b %H:%M} to {end:%a %H:%M}. The model uses glucose up to {known_until:%a %d %b %H:%M} (red on the "
         "timeline). The truth inside the window (green) stays hidden until you press Show truth.")

    truth_on = st.session_state.get("show_truth") and has_truth
    fig = go.Figure()
    tail = all_glucose[(all_glucose.index >= start - pd.Timedelta(hours=6)) & (all_glucose.index < start)]
    line(fig, tail.index, smooth(tail), "known glucose", ROLE["known"], 2.2)
    for k in kinds:
        p = preds[k]
        add_band(fig, p.index, p["pred"], p["sd"], ROLE[k], f"{KINDS[k]} 80 % band")
        line(fig, p.index, p["pred"], KINDS[k], ROLE[k], 2.8, dash="dash")
    if truth_on:
        v = validation["glucose_raw"][(validation.index >= start) & (validation.index <= end)]
        line(fig, v.index, smooth(v), "truth", ROLE["truth"], 2.8)
    mark_events(fig, start - pd.Timedelta(hours=3), end + pd.Timedelta(hours=1))
    if mode == "After a meal":
        for mins in (30, 60, 90, 120):
            t_ = start + pd.Timedelta(minutes=mins)
            fig.add_vline(x=t_, line=dict(color=P["tx3"], width=1))
            fig.add_annotation(x=t_, y=0, yref="paper", yanchor="top", showarrow=False, text=f"+{mins}",
                               font=dict(size=11, color=P["tx2"]))
    if backend == "t1d":
        ins = series[(series["bolus"] > 0) & (series.index >= start - pd.Timedelta(hours=3)) & (series.index <= end)]
        fig.add_scatter(x=ins.index, y=[50] * len(ins), mode="markers+text", text=[f"{u:.1f} U" for u in ins["bolus"]],
                        hovertemplate="%{text}", textposition="top center",
                        marker=dict(color=ROLE["insulin"], symbol="triangle-up", size=10), name="insulin")
    show(fig, 520, yaxis_title="mg/dL", xaxis_range=[start - pd.Timedelta(hours=3), end + pd.Timedelta(minutes=20)])
    note("Dotted yellow lines: meals, including ones just outside the window. A meal the model was not told about "
         "(or one right after the window) can explain part of a miss.")

    if truth_on:
        parts = []
        for k in kinds:
            p = preds[k]
            both = pd.concat([p["pred"], p["sd"], validation["glucose_raw"]], axis=1, keys=["p", "sd", "a"]).dropna()
            if len(both):
                parts.append(f"<span style='color:{ROLE[k]}'><b>{KINDS[k]}</b></span> error "
                             f"<span class='big' style='color:{ROLE[k]}'>{np.sqrt(np.mean((both.p - both.a) ** 2)):.1f}</span> mg/dL RMSE, "
                             f"truth inside the 80 % band {(abs(both.p - both.a) <= Z80 * both.sd).mean():.0%} of the time")
        st.markdown("<br>".join(parts), unsafe_allow_html=True)
    rows = []
    for m in meal_list:
        t = m["datetime"].floor("5min")
        if not start <= t <= end:
            continue
        r = {"meal": meal_label(m), "carbs (g)": m["carbs"], "kcal": m["calories"]}
        window = [t + pd.Timedelta(minutes=5 * j) for j in range(25)]
        for k in kinds:
            p = preds[k]
            w, wsd = p["pred"].reindex(window), p["sd"].reindex(window)
            if w.notna().sum() > 20:
                j = int(np.nanargmax(w.to_numpy()))
                g0 = w.iloc[0] if np.isfinite(w.iloc[0]) else series["glucose"].get(t - pd.Timedelta(minutes=5), np.nan)
                r[f"{KINDS[k]} spike"] = f"{w.iloc[j] - g0:.0f} ± {wsd.iloc[j]:.0f}"
                r[f"{KINDS[k]} peak at"] = f"{j * 5} min"
        if truth_on and validation["glucose_raw"].reindex(window).notna().sum() > 20:
            w = validation["glucose_raw"].reindex(window)
            r["true spike"], r["true peak at"] = round(w.max() - w.iloc[0], 1), f"{int(np.nanargmax(w.to_numpy())) * 5} min"
        rows.append(r)
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    dl = pd.concat(preds, axis=1)
    dl.columns = [f"{k}_{c}" for k, c in dl.columns]
    st.download_button("Download the prediction", dl.round(1).rename_axis("time").to_csv(), "prediction.csv")

with tab_pred:
    if tab_pred.open:
        predict_panel()

# ---------- Decomposition ----------
@st.cache_data(show_spinner="Decomposing the prediction...", max_entries=32)
def decomposition(_m, _post, key, kind_, start_, end_, meal_time):
    """Everything the decomposition plots need, cached on the plan, window, model and chosen meal."""
    series_ = build_series(start_)
    p_, psd_ = predict_window(_m, series_[COLUMNS], _post, start_, end_, base)
    s_ = series_[COLUMNS].copy()
    s_.loc[p_.index, "glucose"] = s_.loc[p_.index, "glucose"].fillna(p_)
    a0_ = s_.index.get_loc(start_.floor("5min"))
    idx_ = s_.index.get_indexer(p_.index)
    V_ = decompose.overlay_terms(_m, s_, a0_, idx_) if kind_ == "overlay" else decompose.regressor_terms(_m, s_, a0_, idx_)
    mean_, sd_ = decompose.evaluate(_m, _post, V_, a0_, idx_)
    meal_ = None
    if kind_ == "overlay" and meal_time is not None:
        step_ = s_.index.get_loc(meal_time.floor("5min"))
        meal_ = decompose.evaluate(_m, _post, decompose.meal_parts(_m, s_, step_, idx_), a0_, idx_)
    return p_, psd_, mean_, sd_, meal_


@st.fragment
def decomposition_panel():
    k = st.radio("Model", list(KINDS), index=1, format_func=KINDS.get, horizontal=True, key="dec_model")
    m = models[k]
    start, end = st.session_state.window
    in_window = [x for x in meal_list if start <= x["datetime"] <= end]
    labels = [meal_label(x) for x in in_window]
    if labels and st.session_state.get("dec_meal") not in labels:
        st.session_state.dec_meal = labels[0]
    sel = in_window[labels.index(st.session_state.dec_meal)] if labels else None
    moved, clicked = timeline("tl_dec", lo, hi, (start, end), glucose_runs, meal_dots, TL_COLORS,
                              selected=sel["datetime"] if sel else None)
    if moved is not None:
        st.session_state.window = moved
        st.rerun()
    if clicked is not None:
        hit = [x for x in in_window if abs((x["datetime"] - clicked).total_seconds()) < 60]
        if hit:
            st.session_state["_set_dec_meal"] = meal_label(hit[0])
            st.rerun()
    note("Same window as Predict. Clicking a meal here only picks which meal's spikes and cooldowns are shown in plot 4.")
    p, psd, mean, sd, meal_parts_ = decomposition(m, post[k], SIG, k, start, end, sel["datetime"] if sel else None)
    times = p.index
    pred, pred_sd = p.to_numpy(), psd.to_numpy()
    residue, residue_sd = mean["residue"].to_numpy(), sd["residue"].to_numpy()
    base_line, base_sd = pred - residue, np.sqrt(pred_sd ** 2 + residue_sd ** 2)
    view_range = [start - pd.Timedelta(minutes=20), end + pd.Timedelta(minutes=20)]

    st.markdown("##### 1. Prediction")
    fig = go.Figure()
    add_band(fig, times, pred, pred_sd, ROLE[k], "80 % band")
    line(fig, times, pred, f"{KINDS[k]} prediction", ROLE[k], 3)
    mark_events(fig, start, end)
    show(fig, 320, yaxis_title="mg/dL", xaxis_range=view_range)

    st.markdown("##### 2. Prediction = baseline + residue")
    note("Baseline: where glucose would go without the events from the window start on (trend, pull back to the usual "
         "level, daily rhythm). Residue: what the events add or take away. Baseline + residue = the prediction above.")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    add_band(fig, times, base_line, base_sd, P["tx2"], "baseline band")
    line(fig, times, base_line, "baseline", P["tx2"], 2.4)
    add_band(fig, times, residue, residue_sd, P["magenta"], "residue band", secondary_y=True)
    line(fig, times, residue, "residue (right axis)", P["magenta"], 2.4, secondary_y=True)
    fig.update_yaxes(title_text="baseline, mg/dL", secondary_y=False)
    fig.update_yaxes(title_text="residue, mg/dL", secondary_y=True, showgrid=False)
    mark_events(fig, start, end)
    show(fig, 320, xaxis_range=view_range)

    if k == "overlay":
        st.markdown("##### 3. Residue = spikes + cooldowns")
        note("Every event (meals, insulin, exercise, quiet time) adds 3 rising curves (spikes) and 3 falling ones "
             "(cooldowns). Here they are summed over every event in and just before the window.")
        fig = go.Figure()
        add_band(fig, times, residue, residue_sd, P["magenta"], "residue band")
        line(fig, times, residue, "residue", P["magenta"], 2.8)
        for n_, color in [("spikes", P["red"]), ("cooldowns", P["blue"])]:
            add_band(fig, times, mean[n_].to_numpy(), sd[n_].to_numpy(), color, f"{n_} band")
            line(fig, times, mean[n_].to_numpy(), n_, color, 2.2)
        fig.add_hline(y=0, line=dict(color=P["tx3"], width=1))
        mark_events(fig, start, end)
        show(fig, 320, yaxis_title="mg/dL", xaxis_range=view_range)

        st.markdown("##### 4. One meal: 3 spikes and 3 cooldowns")
        if sel is None:
            note("No meal inside the window. Drag the timeline dots to include one.")
        else:
            st.selectbox("Meal", labels, key="dec_meal")
            mm, ms_ = meal_parts_
            fig = go.Figure()
            for names_, colors_ in [(["spike 1", "spike 2", "spike 3"], SPIKE_COLORS),
                                    (["cooldown 1", "cooldown 2", "cooldown 3"], COOL_COLORS)]:
                for n_, color in zip(names_, colors_):
                    add_band(fig, times, mm[n_].to_numpy(), ms_[n_].to_numpy(), color, f"{n_} band")
                    line(fig, times, mm[n_].to_numpy(), n_, color, 2.2, dash=None if n_.startswith("spike") else "dot")
            fig.add_hline(y=0, line=dict(color=P["tx3"], width=1))
            fig.add_vline(x=sel["datetime"], line=dict(color=P["yellow"], dash="dot", width=1.2))
            show(fig, 360, yaxis_title="mg/dL", xaxis_range=view_range)
            note(f"{meal_label(sel)}: spikes sum to {mm['all spikes'].iloc[-1]:+.0f} mg/dL and cooldowns to "
                 f"{mm['all cooldowns'].iloc[-1]:+.0f} mg/dL by the end of the window. Solid: spikes, dotted: cooldowns. "
                 "Each is a sigmoid with its own delay and steepness, scaled by this meal and this person; the sign is "
                 "included, so a cooldown that ends up positive means the model learned it pulls the other way.")
    else:
        st.markdown("##### 3. Residue, split by input")
        note("The regressor has no spikes or cooldowns: each input (carbs, time of day, exercise, ...) has its own effect curve.")
        fig = go.Figure()
        line(fig, times, residue, "residue", P["magenta"], 2.8)
        for j, n_ in enumerate([c for c in mean.columns if c != "residue" and mean[c].abs().max() > 0.5]):
            add_band(fig, times, mean[n_].to_numpy(), sd[n_].to_numpy(), PART_COLORS[j % len(PART_COLORS)], f"{n_} band")
            line(fig, times, mean[n_].to_numpy(), n_, PART_COLORS[j % len(PART_COLORS)], 1.8)
        fig.add_hline(y=0, line=dict(color=P["tx3"], width=1))
        mark_events(fig, start, end)
        show(fig, 380, yaxis_title="mg/dL", xaxis_range=view_range)

    with st.expander("Where the bands come from", expanded=False):
        st.markdown("""
Everything starts from how sure the model is about this person's own numbers (their coefficients).

- **The person's fit.** Their history is fitted by least squares, pulled toward their group. What is left over
  (the residual) gives the noise level, the mean squared residual per point (like a chi-squared per point).
  Together with how much data there is and how strongly it is pulled to the group, that gives the
  coefficients' covariance: noise level x (data + pull)^-1.
- **Bottom: each spike and cooldown (plot 4).** Each one is a fixed curve times a sum of the person's
  coefficients, so its uncertainty follows directly from that covariance.
- **Sums (plots 3 and 2).** Spikes, cooldowns and the residue are sums of those pieces. Their bands are
  computed from the summed curve, which keeps the correlations between pieces. That is why a sum's band
  is not just the pieces' bands added up.
- **Top: the prediction (plot 1).** Its band adds the noise level on top of the coefficient uncertainty.
  For long windows the forecast is rolled forward in 4 hour steps, and the uncertainty of each new
  starting point is carried into the next step.
- **Calibration.** Points 5 minutes apart around the same meal are strongly related, so the raw numbers
  are too confident. On days the model never saw (days 6 and 7), every band was scaled until the scaled
  errors had a chi-squared per point of 1. On the separate test days, the 80 % band then held the truth
  81 to 86 % of the time (metabolic) and 87 to 88 % (type 1).
- **Baseline (plot 2).** It is the prediction minus the residue. Its band combines the two as if they
  were independent, so it is an upper bound.
- **Not included:** the uncertainty of the group-level numbers (fitted on many people, so small) and of
  the overlay shapes' delays and steepness, which are treated as fixed.
""")

with tab_dec:
    if tab_dec.open:
        decomposition_panel()

# ---------- Accuracy ----------
with tab_acc:
    st.markdown("##### How well do the models predict people they have never seen?")
    note("Test: each patient's days 1 to 7 are given to the model, and it predicts every meal after that. The patient was "
         "never used to train the model. Errors are measured 30, 60, 90 and 120 minutes after each meal, against the raw sensor.")
    if (RESULTS / "scores.csv").exists():
        sc = pd.read_csv(RESULTS / "scores.csv")
        sc = sc[(sc["backend"] == backend)]
        person = sc[sc["variant"] == "person"].set_index("kind")
        flat = sc[sc["variant"] == "flat"].set_index("kind")
        parts = []
        for k_ in KINDS:
            if k_ in person.index:
                r = person.loc[k_]
                cov = f", truth inside the 80 % band {r['cov60']:.0f} % of the time at 60 min" if "cov60" in r and pd.notna(r["cov60"]) else ""
                parts.append(f"<span style='color:{ROLE[k_]}'><b>{KINDS[k_]}</b></span>: typical error "
                             f"<span class='big' style='color:{ROLE[k_]}'>{r['rmse_all']:.0f}</span> mg/dL "
                             f"(a flat line that ignores meals: {flat.loc[k_, 'rmse_all']:.0f}){cov}.")
        st.markdown("<br>".join(parts) + f"<br><span class='muted'>{int(person['meals'].iloc[0])} test meals, "
                    f"{'metabolic' if backend == 'metabolic' else 'type 1'} model.</span>", unsafe_allow_html=True)
        with st.expander("Full table"):
            st.dataframe(sc.set_index(["kind", "variant"])[[c for c in sc.columns if c.startswith(("rmse_", "top_spike", "cov", "meals"))]], width="stretch")
            note("population / group / person: the same model knowing nothing about the person, knowing their type, and "
                 "knowing their 7 days. top_spike: each day's largest spike (bias below 0 means predicted too small). "
                 "cov: % of test points inside the 80 % band.")
    if (RESULTS / "meal_by_meal.png").exists():
        st.markdown("##### Does it learn a person over time?")
        note("Error on a new person's next meal, as the model sees more of their meals.")
        st.image(str(RESULTS / "meal_by_meal.png"))
    if (RESULTS / "versions.csv").exists():
        st.markdown("##### Did each improvement help?")
        vr = pd.read_csv(RESULTS / "versions.csv")
        st.dataframe(vr[vr["backend"].str.startswith(backend)].set_index(["backend", "kind", "version"])[
            ["rmse_all", "rmse_err30", "rmse_err60", "rmse_err90", "rmse_err120", "top_spike_bias"]].round(1), width="stretch")
        note("v1: meals only. v2: plus cleaning, meal number, food eaten earlier, exercise, glucose before the meal, own past "
             "spikes. v3: plus calories, clock time, time since last meal, usual level, swings, very large meals, meal "
             "speed (overlay). v4: plus exercise and quiet-time events.")
    kap = {k_: np.asarray(getattr(models[k_], "kappa", np.ones(48))) for k_ in KINDS}
    if any(np.any(v != 1) for v in kap.values()):
        with st.expander("Band calibration"):
            st.dataframe(pd.DataFrame({KINDS[k_]: np.round(v[[5, 11, 17, 23]], 2) for k_, v in kap.items()},
                                      index=["30 min", "60 min", "90 min", "120 min"]), width="stretch")
            note("The raw uncertainty is multiplied by these factors, set on days 6 and 7, so the bands match real errors.")
    if getattr(models["overlay"], "warp", False):
        with st.expander("What makes a meal's spike faster or slower (overlay model)"):
            st.dataframe(pd.DataFrame({"factor (above 1 = slower)": np.exp(models["overlay"].warp_w).round(2)}, index=WARP_FEATURES), width="stretch")
