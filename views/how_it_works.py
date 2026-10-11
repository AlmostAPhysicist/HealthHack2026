"""How it works: one real question (Sam's meal), answered step by step, with the tools and data behind each step."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style import ROLES, TYPE_NAME, brand, css, flow, photo_file

css()
st.markdown("""<style>
  .title {font-size: 3rem; line-height: 1.05; font-weight: 700; margin: 1rem 0 .4rem;}
  .part {font-size: .9rem; letter-spacing: .12em; text-transform: uppercase; color: var(--magenta); margin-top: 3.2rem;}
  .part-title {font-size: 2.1rem; line-height: 1.15; margin: .1rem 0 .5rem;}
  .q {font-size: 2rem; line-height: 1.3; max-width: 40rem;}
  .pts {margin: .4rem 0 .8rem; padding-left: 1.1rem;}
  .pts li {font-size: 1.06rem; line-height: 1.5; margin: .2rem 0;}
  .kv {display: grid; grid-template-columns: max-content max-content 1fr; gap: .25rem 1.2rem; font-size: 1.02rem; margin: .4rem 0;}
  .kv .k {color: var(--tx2);}
  .refs li {font-size: .88rem; color: var(--tx2); margin: .2rem 0;}
</style>""", unsafe_allow_html=True)

REFS = [
    ("Zeevi D, et al. Personalized nutrition by prediction of glycemic responses. Cell, 2015.", "https://doi.org/10.1016/j.cell.2015.11.001"),
    ("Bell KJ, et al. Impact of fat, protein, and glycemic index on postprandial glucose control in type 1 diabetes. "
     "Diabetes Care, 2015.", "https://doi.org/10.2337/dc15-0100"),
    ("Colberg SR, et al. Physical activity/exercise and diabetes: a position statement of the American Diabetes "
     "Association. Diabetes Care, 2016.", "https://doi.org/10.2337/dc16-1728"),
    ("Wolever TMS, et al. Second-meal effect: low-glycemic-index foods eaten at dinner improve subsequent breakfast "
     "glycemic response. Am J Clin Nutr, 1988.", "https://doi.org/10.1093/ajcn/48.4.1041"),
    ("Monnier L, et al. Magnitude of the dawn phenomenon and its impact on the overall glucose exposure in type 2 "
     "diabetes. Diabetes Care, 2013.", "https://doi.org/10.2337/dc13-0749"),
    ("Radford A, et al. Learning transferable visual models from natural language supervision (CLIP). ICML, 2021.",
     "https://arxiv.org/abs/2103.00020"),
    ("OpenAPS oref0: open-source insulin-on-board and dosing maths.", "https://github.com/openaps/oref0"),
    ("Székely GJ, Rizzo ML, Bakirov NK. Measuring and testing dependence by correlation of distances. Ann Stat, 2007. "
     "Python: dcor.", "https://github.com/vnmabus/dcor"),
    ("CGMacros: glucose, meals, photos and activity of 45 people (PhysioNet).", "https://physionet.org/content/cgmacros/"),
    ("USDA FNDDS, the food and nutrient database for the US national diet survey (FoodData Central).", "https://fdc.nal.usda.gov/"),
    ("MetaboNet: merged public type 1 diabetes studies.", "https://metabo-net.org"),
    ("IDF Diabetes Atlas, 2025.", "https://diabetesatlas.org"),
]
ref = lambda *n: "<span class='ref'>[" + ",".join(f"<a href='#ref-{k}'>{k}</a>" for k in n) + "]</span>"
link = lambda text, url: f"<a href='{url}' target='_blank'>{text}</a>"
pts = lambda items: st.markdown("<ul class='pts'>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>", unsafe_allow_html=True)

brand("2.4rem")
st.markdown("<div class='title'>How it works</div>", unsafe_allow_html=True)

from engine import (ROOT, AFTER, COLUMNS, learn, palette, plan_series, predict_after_meal, predict_window, read, rgba,  # noqa: E402
                    show, track_record)

P = palette()
C = {r: P[c] for r, (c, _) in ROLES.items()}
DEMO = ROOT / "samples" / "home_demo" / "Sam"
about = pd.read_csv(DEMO / "about_me.csv").iloc[0]
history = read(DEMO / "history.csv")
meal_row = pd.read_csv(DEMO / "meal.csv").iloc[0]
t0 = pd.Timestamp(meal_row["time"])
MEAL = str(meal_row["meal"]).lower()                  # breakfast, lunch, ...
meal = {"time": t0, "name": meal_row["meal"], **{c: float(meal_row[c]) for c in ["carbs", "protein", "fat", "fiber", "calories"]}}
truth = pd.read_csv(DEMO / "truth.csv", parse_dates=["time"]).set_index("time")["glucose"]
explain = json.loads((ROOT / "results" / "explain.json").read_text()) if (ROOT / "results" / "explain.json").exists() else {}
scores = pd.read_csv(ROOT / "results" / "scores.csv") if (ROOT / "results" / "scores.csv").exists() else None
BACKEND = "t1d" if about["diabetes_type"] == "t1d" else "metabolic"
s_meal = plan_series(history, t0, t0 + AFTER, BACKEND, [meal])
known = history["glucose_raw"][(history.index >= t0 - pd.Timedelta(hours=2)) & (history.index < t0)]


def sam(kind=None):
    """Learn Sam (only when a part needs it; kept for every visitor)."""
    with st.spinner("Learning how Sam responds to food..."):
        return learn(history, {"group": about["diabetes_type"]}, "demo-Sam", kind=kind, exclude=about["dataset_id"], shared=True)


def line(fig, x, y, name, color, width=2.4, dash=None):
    fig.add_scatter(x=x, y=y, name=name, line=dict(color=color, width=width, dash=dash), hovertemplate="%{y:.0f}")


TOTAL = 9
st.session_state.setdefault("hiw_step", 0)
if st.toggle("Show every part at once", key="hiw_all"):
    st.session_state.hiw_step = TOTAL


def part(n, title):
    st.markdown(f"<div class='part'>Part {n}</div><div class='part-title'>{title}</div>", unsafe_allow_html=True)


def stop_here(n, label):
    """Reveal the next part on request, so nothing below is computed until someone wants it."""
    if st.session_state.hiw_step > n:
        return False
    if st.button(f"Next: {label}", key=f"next_{n}", type="primary"):
        st.session_state.hiw_step = n + 1
        st.rerun()
    return True


# ---------- The question ----------
c1, c2 = st.columns([1.2, 1], gap="large", vertical_alignment="center")
c1.markdown(f"<div class='part'>The question</div><div class='q'>Sam wants to know what his glucose will do after this "
            f"{MEAL}.</div><p class='muted' style='margin-top:.8rem'>Sam has type 2 diabetes and is a real, anonymous "
            f"person from the CGMacros study{ref(9)}. The model never trained on Sam's data.</p>", unsafe_allow_html=True)
c2.image(str(DEMO / "meal_photo.jpg"), width="stretch")
if stop_here(0, "getting to know Sam"):
    st.stop()

# ---------- I ----------
part("I", "Getting to know Sam")
c1, c2, c3 = st.columns([1, 1, 1.3], gap="medium")
c1.image(str(ROOT / "static" / "hiw" / "sensor_close.webp"), width="stretch", caption="Glucose sensor. Photo: Raimond Spekking, CC BY-SA 4.0")
c2.image(str(ROOT / "static" / "hiw" / "sensor_arm.webp"), width="stretch", caption="Glucose sensor, X-ray. Photo: Hellerhoff, CC BY-SA 4.0")
c3.markdown("<p style='font-size:1.1rem;line-height:1.6'>Sam wears a glucose sensor on his arm, reading day and night, "
            "and a fitness band for activity and heart rate. He has also been logging every meal, with a photo.</p>",
            unsafe_allow_html=True)
week = history[history.index >= history.index[-1] - pd.Timedelta(days=4)]
fig = go.Figure()
line(fig, week.index, week["glucose_raw"], "glucose", C["glucose"], 1.6)
m_ = week[week["carbs"] > 0]
fig.add_scatter(x=m_.index, y=[week["glucose_raw"].min() - 10] * len(m_), mode="markers", name="meals",
                marker=dict(color=C["food"], size=8), text=[f"{c:.0f} g carbs" for c in m_["carbs"]], hovertemplate="%{text}")
show(fig, 280, yaxis_title="mg/dL")
if stop_here(1, "cleaning and classification"):
    st.stop()

# ---------- II ----------
part("II", "Cleaning and classification")
flow([("Raw data", "glucose"), ("Remove noise: wiggles under 1 % of the level", "model"),
      ("Classify the diabetes type", "model"), ("Pick the model to use", "prediction")])
z = history[(history.index >= t0 - pd.Timedelta(hours=30)) & (history.index < t0 - pd.Timedelta(hours=22))]
from classifier import classify  # noqa: E402
from engine import _classifier  # noqa: E402
guess = classify({}, history, _classifier())
probs = guess["probabilities"] or {}
c1, c2 = st.columns(2, gap="large")
with c1:
    fig = go.Figure()
    line(fig, z.index, z["glucose_raw"], "raw", P["tx3"], 3)
    line(fig, z.index, z["glucose"], "cleaned", C["glucose"], 1.8)
    show(fig, 240, yaxis_title="mg/dL", title="Cleaning")
with c2:
    fig = go.Figure(go.Bar(x=[probs.get(g, 0) for g in TYPE_NAME], y=[TYPE_NAME[g] for g in TYPE_NAME], orientation="h",
                           marker_color=[P["green"], P["yellow"], P["orange"], P["magenta"]], hovertemplate="%{x:.0%}"))
    show(fig, 240, xaxis_tickformat=".0%", xaxis_range=[0, 1], showlegend=False, title="Classifying (from glucose alone)")
    st.markdown("<p class='muted'>A <b class='k-open'>scikit-learn</b> logistic regression learned from people of known "
                "type, using 9 patterns in their glucose (no fixed clinical cut-offs).</p>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size:1.9rem;line-height:1.3;margin:1.4rem 0;max-width:44rem'><b class='c-red'>4 in 10</b> adults "
            f"with diabetes don't know they have it{ref(12)}.</div>", unsafe_allow_html=True)
with st.expander("The two models side by side (type 1 and everyone else)"):
    st.markdown("""<table style='width:100%;font-size:.95rem'>
<tr class='small'><th></th><th style='text-align:left'>Healthy, prediabetes, type 2</th><th style='text-align:left'>Type 1</th></tr>
<tr><td>Maths</td><td colspan='2'>the same two model kinds (Part V), the same code</td></tr>
<tr><td>Insulin</td><td>the body's own, estimated from the carbs through a faster <b class='k-open'>OpenAPS</b> curve (peak 45 min, done in 4 h); the model learns how strongly each person responds</td><td>the doses you plan, through the <b class='k-open'>OpenAPS</b> curve: the prediction shows whether that dose over- or undershoots</td></tr>
<tr><td>Food</td><td>carbs, protein, fat, fiber, calories</td><td>carbs (all those studies log)</td></tr>
<tr><td>Trained on</td><td>45 people, <b class='k-open'>CGMacros</b></td><td>334 people, <b class='k-open'>MetaboNet</b> (48 more for testing)</td></tr>
<tr><td>Learned separately</td><td colspan='2'>every weight, timing and setting</td></tr>
</table>""", unsafe_allow_html=True)
if stop_here(2, "what's in the meal?"):
    st.stop()

# ---------- III ----------
part("III", "What's in the meal?")
ph = explain.get("photo")
c1, c2 = st.columns([1, 1.3], gap="large")
c1.image(str(DEMO / "meal_photo.jpg"), width="stretch")
with c2:
    if ph and "candidates" in ph:
        rows = "".join(f"<div><b class='k-open'>{share:.0%}</b> &nbsp;{f['name']}</div>" for f, share in ph["candidates"])
        st.markdown(f"<div class='small'>CLIP's top 3 guesses:</div><div style='line-height:1.8'>{rows}</div>"
                    f"<p class='small' style='margin-top:.5rem'>Sam logged {meal['carbs']:.0f} g carbs.</p>", unsafe_allow_html=True)
        near = [dict(n, photo=f) for n in ph.get("neighbours", []) if (f := photo_file(n.get("photo")))][:3]
        if near:
            st.markdown("<div class='small'>Logged meals that look the same:</div>", unsafe_allow_html=True)
            cols_ = st.columns(len(near))
            for c_, n in zip(cols_, near):
                c_.image(n["photo"], width="stretch", caption=f"{n['similarity']:.2f} alike · {n['carbs']:.0f} g carbs")
pts([f"<b class='k-open'>CLIP</b>{ref(6)} (open source, OpenAI) looks at the photo and finds the food names that describe "
     "it best, like \"a burrito bowl\".",
     f"Those names alone carry no nutrition, so the 3 best are looked up in <b class='k-open'>USDA FNDDS</b>{ref(10)}, the US "
     "government's food database, which gives nutrients per 100 g and the usual portion.",
     "A database entry is a generic version of the dish, while people tend to eat the same meals cooked their own way. So "
     "we also look for logged meal photos that look like this one, and a close match brings its real numbers instead."])
if stop_here(3, "food and insulin in the blood"):
    st.stop()

# ---------- IV ----------
part("IV", "Food and insulin in the blood")
from exogenous import bump, insulin_used  # noqa: E402
t = np.arange(0, 361, 5)
fig = go.Figure()
for p_, w in ((20, 1.6), (60, 2.4), (150, 1.6)):
    line(fig, t, bump(p_)(t), f"food, peaks at {p_} min", C["food"], w, None if p_ == 60 else "dot")
act = np.gradient(insulin_used()(t), t)
line(fig, t, act / act.max(), "insulin at work (OpenAPS)", C["open"], 2.6)
show(fig, 280, xaxis_title="minutes after eating or dosing", yaxis_title="effect (scaled to 1)")
pts([f"Food reaches the blood over minutes to hours, more slowly with fat, protein and fiber{ref(2)}, so each meal is "
     "spread over time with a few of these shapes, and the model learns how much of each fits Sam.",
     f"Insulin then brings glucose back down. To know how fast, we use <b class='k-open'>OpenAPS</b>{ref(7)}, an open-source "
     "artificial pancreas project. Its formula only says how a dose of rapid insulin acts over time (strongest at about 75 "
     "minutes, used up by about 6 hours); it doesn't look at food or know anyone's diabetes type. So for type 1 we give it "
     "the real doses, and for everyone else, who have no doses, we give it the carbs as a stand-in for the insulin their "
     "body releases, with faster timing, and the model learns how strongly each person responds.",
     "Stress, illness, sleep and hormones change this too, but no sensor records them, so the model can't see them."])
if stop_here(4, "the model"):
    st.stop()

# ---------- V ----------
part("V", "The model")
from context import meal_table  # noqa: E402
from engine import best_kind, load_models, code_version  # noqa: E402
row = meal_table(s_meal[s_meal.index <= t0]).iloc[-1]
g_now = float(known.iloc[-1]) if len(known) else np.nan
fmt = lambda v, f: f.format(v) if pd.notna(v) else "not known yet"
PARAMS = [
    ("food", "Carbs, protein, fat, fiber, calories", f"{meal['carbs']:.0f} / {meal['protein']:.0f} / {meal['fat']:.0f} / {meal['fiber']:.0f} g, {meal['calories']:.0f} kcal"),
    ("food", "Meal of the day; carbs eaten earlier today", f"{['first', 'second', 'third or later'][int(row['slot']) - 1]}; {row['carbs_today']:.0f} g"),
    ("food", "Time since the last meal" + ref(4), f"{row['mins_since_meal'] / 60:.1f} h"),
    ("activity", "Activity, heart rate, exercise" + ref(3), f"{row['exercise_2h']:.0f} above usual (2 h before)"),
    ("glucose", "Glucose now, trend, acceleration", f"{g_now:.0f} mg/dL, {row['pre_vs_normal']:+.0f} vs usual"),
    ("model", "Sam's usual level, swings, daily peak and low", f"{fmt(row['usual_level'], '{:.0f}')} mg/dL, ± {row['swing']:.0f}, +{row['daily_peak']:.0f} / -{row['daily_low']:.0f}"),
    ("model", "Sam's past rise per 10 g carbs", fmt(row["past_rise_per_10g"], "{:.1f} mg/dL")),
    ("model", "Time of day" + ref(5), f"{t0:%H:%M}"),
]
with st.expander(f"What the model reads at Sam's {MEAL} (all from before the meal)"):
    st.markdown("<div class='kv'>" + "".join(f"<div class='k-{r}'>&#9679;</div><div class='k'>{k}</div><div>{v}</div>"
                                             for r, k, v in PARAMS) + "</div>", unsafe_allow_html=True)
dc = explain.get("distance_correlation")
if dc:
    c1, c2 = st.columns(2, gap="large")
    for col, (b, title) in zip((c1, c2), (("metabolic", "Healthy, prediabetes, type 2"), ("t1d", "Type 1"))):
        d = dict(list(dc.get(b, {}).items())[:8])
        if d:
            with col:
                fig = go.Figure(go.Bar(x=list(d.values())[::-1], y=list(d.keys())[::-1], orientation="h", marker_color=C["model"],
                                       hovertemplate="%{x:.2f}"))
                show(fig, 290, title=title, showlegend=False, xaxis_range=[0, max(0.5, max(d.values()) + .05)])
    st.markdown(f"<p class='muted'>How strongly each value goes with the rise after a meal, over all our meals. "
                f"<b class='k-open'>Distance correlation</b>{ref(8)} (Python package dcor) is 0 for unrelated and 1 for fully "
                "determined, and also catches curved relationships.</p>", unsafe_allow_html=True)

st.markdown("<div class='part-title' style='font-size:1.6rem;margin-top:1.4rem'>Model 1: the regressor</div>", unsafe_allow_html=True)
flow([("Glucose now", "glucose"), ("Each value's curve x its weight", "model"), ("Add them up", "model"),
      ("Glucose at 5 to 240 min", "prediction")])
person = sam("regressor")
model, coef = person["model"], person["post"][0]
pts(["Every past moment of Sam's data shows where glucose went next, so the weights are set to miss those moments least "
     "(least squares).",
     "A week is only a few dozen meals, so the weights start from everyone's and move toward Sam's own only as far as "
     "Sam's data supports. That much data is also too little for a neural network, which is why the model is linear, and "
     "being linear means every part can be shown below."])
anchor = int(np.flatnonzero(s_meal.index < t0)[-1])
parts_, _ = model.components(s_meal[COLUMNS], anchor, person["post"], person["base"])
role_of = lambda n: ("food" if any(w in n for w in ("carb", "meal", "protein", "fiber", "calor", "fat", "past spikes", "eaten"))
                     else "insulin" if "insulin" in n else "activity" if any(w in n for w in ("activ", "exercise", "heart", "quiet"))
                     else "glucose" if any(w in n for w in ("trend", "pull")) else "model")


def spark(v, color):
    v = np.asarray(v, float)
    m = max(np.abs(v).max(), 1e-9)
    pts_ = " ".join(f"{i * 120 / (len(v) - 1):.1f},{14 - val / m * 12:.1f}" for i, val in enumerate(v))
    return (f"<svg viewBox='0 0 120 28' width='120' height='28'><line x1='0' y1='14' x2='120' y2='14' stroke='var(--tx3)' "
            f"stroke-dasharray='2 3'/><polyline points='{pts_}' fill='none' stroke='var(--{color})' stroke-width='2'/></svg>")


with st.expander(f"Every part of the regressor for Sam's {MEAL}: its push (mg/dL) and its shape"):
    curve = parts_.loc[parts_.index <= 180]
    names = np.array([model.group_of(f) for f in model.features])
    rows_html = ""
    for n in curve.abs().max().sort_values(ascending=False).index:
        v = curve[n]
        push = lambda m_: f"<span class='{'k-glucose' if v.get(m_, 0) > 0.5 else 'k-model' if v.get(m_, 0) < -0.5 else 'small'}'>{v.get(m_, 0):+.1f}</span>"
        rows_html += (f"<tr><td><span class='k-{role_of(n)}'>&#9679;</span> {n}</td><td>{(names == n).sum()}</td><td>{push(30)}</td>"
                      f"<td>{push(60)}</td><td>{push(120)}</td><td>{spark(v.to_numpy(), ROLES[role_of(n)][0])}</td></tr>")
    st.markdown("<table style='width:100%;font-size:.95rem'><tr class='small'><th style='text-align:left'>part</th><th>curves</th>"
                "<th>30 min</th><th>60 min</th><th>120 min</th><th>shape, 3 h</th></tr>" + rows_html + "</table>",
                unsafe_allow_html=True)
    grp_c = model.prior(person["fit_group"], exclude=about["dataset_id"] if about["dataset_id"] in getattr(model, "person_stats", {}) else None)
    st.dataframe(pd.DataFrame({"feature": model.features, "part": names, "everyone": model.population.round(4),
                               TYPE_NAME[person["group"]]: grp_c.round(4), "Sam": coef.round(4)}), hide_index=True, width="stretch")

st.markdown("<div class='part-title' style='font-size:1.6rem;margin-top:1.4rem'>Model 2: the overlay</div>", unsafe_allow_html=True)
flow([("Baseline: trend and usual level", "glucose"), ("+ 3 rising S-curves per event", "food"),
      ("+ 3 falling S-curves per event", "insulin"), ("= prediction", "prediction")])
st.markdown("<p class='muted'>Real curves often have double peaks, plateaus, delays and dips, which fixed bumps struggle "
            "to draw. Adding S-curves with their own timing can draw them, and each meal also gets its own speed.</p>",
            unsafe_allow_html=True)
try:
    from decompose import evaluate, meal_parts, overlay_terms
    ov = sam("overlay")
    om, opost = ov["model"], ov["post"]
    s_ = s_meal[COLUMNS]
    idx = np.arange(anchor + 1, anchor + 1 + 48)                          # 4 h, so the dip after the peak shows
    times = s_.index[idx]
    s_ = plan_series(history, t0, t0 + pd.Timedelta(hours=4), BACKEND, [meal])[COLUMNS]
    pred_o, _ = predict_window(om, s_, opost, t0, t0 + pd.Timedelta(hours=4), ov["base"])
    terms, _ = evaluate(om, opost, overlay_terms(om, s_, anchor, idx), anchor, idx)
    step = int(np.flatnonzero(s_.index == t0.floor("5min"))[0])
    mparts, _ = evaluate(om, opost, meal_parts(om, s_, step, idx), anchor, idx)
    pred_o = pred_o.reindex(times)
    levels = ["prediction = baseline + change from events", "change = spikes + cooldowns",
              "this meal's spikes = 1 + 2 + 3", "this meal's cooldowns = 1 + 2 + 3"]
    lvl = st.segmented_control("Break it down", levels, default=levels[0], key="hiw_decomp")
    fig = go.Figure()
    mins = (times - t0).total_seconds() / 60
    if lvl == levels[0]:
        line(fig, mins, pred_o.to_numpy(), "prediction", C["prediction"], 3)
        line(fig, mins, pred_o.to_numpy() - terms["residue"].to_numpy(), "baseline", C["glucose"], 2, "dot")
        line(fig, mins, terms["residue"].to_numpy(), "change from events", C["model"], 2, "dash")
        line(fig, mins, truth.reindex(times).to_numpy(), "what really happened", C["truth"], 2)
    elif lvl == levels[1]:
        line(fig, mins, terms["residue"].to_numpy(), "change from events", C["model"], 3)
        line(fig, mins, terms["spikes"].to_numpy(), "spikes", C["food"], 2)
        line(fig, mins, terms["cooldowns"].to_numpy(), "cooldowns", C["insulin"], 2)
    else:
        kind_ = "spike" if "spikes" in lvl else "cooldown"
        col_ = C["food" if kind_ == "spike" else "insulin"]
        line(fig, mins, mparts[f"all {kind_}s"].to_numpy(), f"all {kind_}s", col_, 3)
        for k_, dash in zip((1, 2, 3), ("dot", "dash", "dashdot")):
            line(fig, mins, mparts[f"{kind_} {k_}"].to_numpy(), f"{kind_} {k_}", col_, 1.6, dash)
    show(fig, 320, xaxis_title=f"minutes after {MEAL}", yaxis_title="mg/dL")
except Exception as e:                                                         # older weights without this model
    st.caption(f"(overlay breakdown unavailable: {e})")

bundle, _ = load_models(code_version())
tr_ = {k: getattr(m, "tune_rmse", None) for (b, k), m in bundle.items() if b == BACKEND}
st.markdown(f"<p>The site uses whichever made smaller errors on held-out days of the training people: the "
            f"<b>{best_kind(bundle, BACKEND)}</b>"
            + (f" ({', '.join(f'{k} {v:.1f}' for k, v in tr_.items())} mg/dL)" if all(tr_.values()) else "") + ".</p>",
            unsafe_allow_html=True)

if stop_here(5, "how sure are we?"):
    st.stop()

# ---------- VI ----------
part("VI", "How sure are we?")
flow([("Replay Sam's past meals", "model"), ("Note each miss", "truth"), ("Add 10 misses from others", "open"),
      ("Band: where 8 in 10 fell", "prediction")])
person = sam()
pred = predict_after_meal(person, history, meal)
fig = go.Figure()
line(fig, known.index, known, f"Sam before {MEAL}", C["glucose"], 2.4)
fig.add_scatter(x=pred.index, y=pred["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=pred.index, y=pred["lo"], fill="tonexty", line=dict(width=0), fillcolor=rgba(C["prediction"], 0.16),
                name="80 % band", hoverinfo="skip")
line(fig, pred.index, pred["pred"], "prediction", C["prediction"], 2.6, "dash")
line(fig, truth.index, truth, "what really happened", C["truth"], 2.6)
show(fig, 320, yaxis_title="mg/dL")
track = track_record(person, history)
pts(["To see how far off we tend to be for Sam, each of Sam's last 20 meals is predicted again from only the data "
     "before it, and the miss is noted every 5 minutes for 3 hours"
     + (f"; their average is the \"off by\" number ({track['mae']:.0f} mg/dL)." if track else "."),
     "Those misses then set the band: it covers where 8 in 10 of them fell. Twenty meals are few, so 10 misses from other "
     "people are mixed in to keep it from being overconfident. Of the bands we tested this was the tightest that still "
     "held 80 % on new meals (48 mg/dL wide).",
     "For type 1 a person's past misses didn't predict their later ones, so type 1 uses one band for everyone instead."])
if stop_here(6, "validation"):
    st.stop()

# ---------- VII ----------
part("VII", "Validation")
flow([("Learn days 1 to 7", "model"), ("Predict every meal after", "prediction"), ("Compare with the raw sensor", "truth"),
      ("Score", "model")])
pts(["A test only counts if the model has never seen the person, so every person without type 1 (45) is left out of the "
     "training before being tested, and for type 1, 48 people were kept out from the start.",
     "The site follows the same rule: the model only learns from data before the moment it predicts, so if your file runs "
     "past it, that later part is kept for Validate!.",
     "Once validated, that meal or day joins your history, so your next prediction learns from it, and it is saved for "
     "retraining."])
if stop_here(7, "results"):
    st.stop()

# ---------- VIII ----------
part("VIII", "Results")
te = ROOT / "results" / "test_errors.csv"
if te.exists():
    e = pd.read_csv(te)
    best = {b: best_kind(bundle, b) for b in e["backend"].unique()}
    e = e[[best.get(b) == k for b, k in zip(e["backend"], e["kind"])]]
    e = e.assign(sq=(e[["err30", "err60", "err90", "err120"]] ** 2).mean(axis=1))
    by = e.groupby("group").agg(rmse=("sq", lambda x: float(np.sqrt(np.mean(x)))), meals=("sq", "size"), people=("id", "nunique"))
    flat = scores[scores["variant"] == "flat"].groupby("backend")["rmse_all"].first() if scores is not None else pd.Series(dtype=float)
    order = [g for g in ("healthy", "prediabetes", "t2d", "t1d") if g in by.index]
    colors = {"healthy": "green", "prediabetes": "yellow", "t2d": "orange", "t1d": "magenta"}
    fig = go.Figure(go.Bar(x=[TYPE_NAME[g] for g in order], y=[by.loc[g, "rmse"] for g in order],
                           marker_color=[P[colors[g]] for g in order], text=[f"{by.loc[g, 'rmse']:.0f}" for g in order],
                           textposition="outside", hovertemplate="%{y:.1f}"))
    show(fig, 300, yaxis_title="typical miss, 30 to 120 min (mg/dL)", showlegend=False)
    st.markdown("<p class='muted'>Typical miss: how far the real reading usually is from the single predicted line, above "
                "or below (the root mean square at 30, 60, 90 and 120 minutes after every meal of never-seen people). It is "
                "roughly how far the band reaches on each side.</p>", unsafe_allow_html=True)
if stop_here(8, "tools"):
    st.stop()

# ---------- IX ----------
part("IX", "Tools and data")
pts([f"<b class='k-open'>Data:</b> CGMacros{ref(9)}, MetaboNet type 1 studies{ref(11)}, USDA FNDDS{ref(10)}.",
     f"<b class='k-open'>Open tools:</b> CLIP{ref(6)} (through PyTorch, Hugging Face and OpenVINO), the OpenAPS insulin "
     f"curve{ref(7)}, scikit-learn, NumPy, SciPy, pandas, dcor{ref(8)}, Streamlit, Plotly.",
     "<b class='k-model'>Ours:</b> cleaning, the meal values, both models, the three-layer learning, the bands, the photo "
     "memory, the site.",
     "Look: Flexoki colours, the Libron typeface, Gray's Anatomy plates (1918, public domain), sensor photos from Wikimedia "
     "Commons. No paid or closed model anywhere. Glucklich predicts; it does not give medical advice."])
st.markdown("<div class='part' style='color:var(--tx2)'>References</div><ol class='refs'>" +
            "".join(f"<li id='ref-{k}'>{text} {link('link', url)}</li>" for k, (text, url) in enumerate(REFS, 1)) + "</ol>",
            unsafe_allow_html=True)
