"""How it works: follow one real meal (Sam's lunch) through every building block, with the reason each one is there."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style import TYPE_NAME, brand, css

css()
st.markdown("""<style>
  .title {font-size: 3rem; line-height: 1.05; font-weight: 700; margin: 1rem 0 .4rem;}
  .lead {font-size: 1.25rem; color: var(--tx2); max-width: 46rem;}
  .chapter {font-size: .9rem; letter-spacing: .12em; text-transform: uppercase; color: var(--magenta); margin-top: 3rem;}
  .ch-title {font-size: 2rem; line-height: 1.15; margin: .1rem 0 .6rem;}
  .why {border-left: 3px solid var(--yellow); padding: .2rem 0 .2rem 1rem; margin: .8rem 0; color: var(--tx);}
  .why b {color: var(--yellow);}
  .tool {display: inline-block; border: 1px solid var(--ui); border-radius: 999px; padding: .05rem .6rem; margin: .15rem .2rem 0 0;
         font-size: .85rem; color: var(--tx2);}
  .story p, .story li {font-size: 1.08rem; line-height: 1.55;}
  .kv {display: grid; grid-template-columns: max-content 1fr; gap: .25rem 1.2rem; font-size: 1.02rem;}
  .kv .k {color: var(--tx2);}
</style>""", unsafe_allow_html=True)

brand("2.4rem")
st.markdown("<div class='title'>How it works</div>"
            "<div class='lead'>Let's follow one real meal all the way through: Sam's lunch. Sam has type 2 diabetes and "
            "a week of glucose sensor data. Every chapter is one building block, what it does to this meal, and why it "
            "is there.</div>", unsafe_allow_html=True)

from engine import (ROOT, AFTER, COLUMNS, learn, palette, plan_series, predict_after_meal, predict_window, read, rgba,  # noqa: E402
                    show, track_record)

P = palette()
DEMO = ROOT / "demo" / "Sam"
about = pd.read_csv(DEMO / "about_me.csv").iloc[0]
history = read(DEMO / "history.csv")
meal_row = pd.read_csv(DEMO / "meal.csv").iloc[0]
t0 = pd.Timestamp(meal_row["time"])
meal = {"time": t0, "name": meal_row["meal"], **{c: float(meal_row[c]) for c in ["carbs", "protein", "fat", "fiber", "calories"]}}
truth = pd.read_csv(DEMO / "truth.csv", parse_dates=["time"]).set_index("time")["glucose"]
explain = json.loads((ROOT / "results" / "explain.json").read_text()) if (ROOT / "results" / "explain.json").exists() else {}
scores = pd.read_csv(ROOT / "results" / "scores.csv") if (ROOT / "results" / "scores.csv").exists() else None

BACKEND = "t1d" if about["diabetes_type"] == "t1d" else "metabolic"


def sam():
    """Learn Sam (only once a chapter needs it; kept for every visitor)."""
    with st.spinner("Learning how Sam responds to food..."):
        return learn(history, {"group": about["diabetes_type"]}, "demo-Sam", kind=about["model"],
                     exclude=about["dataset_id"], shared=True)

TOTAL = 11
st.session_state.setdefault("hiw_step", 1)
if st.toggle("Show every chapter at once", key="hiw_all"):
    st.session_state.hiw_step = TOTAL


def chapter(n, title, tools=()):
    st.markdown(f"<div class='chapter'>Chapter {n} of {TOTAL}</div><div class='ch-title'>{title}</div>"
                + "".join(f"<span class='tool'>{t}</span>" for t in tools), unsafe_allow_html=True)


def why(text):
    st.markdown(f"<div class='why'><b>Why:</b> {text}</div>", unsafe_allow_html=True)


def next_button(n, label):
    """Reveal the next chapter. Returns True if the story should stop here."""
    if st.session_state.hiw_step > n:
        return False
    if st.button(f"Next: {label}", key=f"next_{n}", type="primary"):
        st.session_state.hiw_step = n + 1
        st.rerun()
    return True


def line(fig, x, y, name, color, width=2.4, dash=None):
    fig.add_scatter(x=x, y=y, name=name, line=dict(color=color, width=width, dash=dash), hovertemplate="%{y:.0f}")


# ---------- 1 ----------
chapter(1, "Sam's lunch")
c1, c2 = st.columns([1, 1.3], gap="large")
c1.image(str(DEMO / "meal_photo.jpg"), width="stretch")
c2.markdown(f"<div class='story'><p>{t0:%A %d %B %Y}, {t0:%H:%M}. Sam takes a photo of lunch and logs it. Behind the "
            f"photo: <b>{meal['carbs']:.0f} g carbs</b>, {meal['protein']:.0f} g protein, {meal['fat']:.0f} g fat, "
            f"{meal['fiber']:.0f} g fiber, {meal['calories']:.0f} kcal.</p>"
            "<p>The question we answer: <b>what will Sam's glucose do over the next three hours?</b> Not an average "
            "person's. Sam's.</p><p>Below, we take this one meal through every step, using Sam's real data. Nothing here "
            "is staged: Sam is a real (anonymous) person from the CGMacros study, and the model has never seen Sam's data "
            "in training.</p></div>", unsafe_allow_html=True)
if next_button(1, "line up a week of data"):
    st.stop()

# ---------- 2 ----------
chapter(2, "A week of data, on one clock", ["pandas", "CGMacros (PhysioNet)"])
week = history[history.index >= history.index[-1] - pd.Timedelta(days=7)]
fig = go.Figure()
line(fig, week.index, week["glucose_raw"], "glucose", P["red"], 1.6)
m = week[week["carbs"] > 0]
fig.add_scatter(x=m.index, y=[week["glucose_raw"].min() - 10] * len(m), mode="markers", name="meals",
                marker=dict(color=P["yellow"], size=8), text=[f"{c:.0f} g carbs" for c in m["carbs"]], hovertemplate="%{text}")
show(fig, 300, yaxis_title="mg/dL")
st.markdown(f"<div class='story'><p>Sam wore a glucose sensor (a reading every 15 minutes), a fitness band (activity, "
            f"heart rate) and logged every meal with a photo. That is {len(week['glucose_raw'].dropna())} glucose readings "
            f"and {len(m)} meals in the last week alone.</p><p>Everything is put on one 5-minute clock: glucose and heart "
            "rate are averaged into each 5-minute step, food, insulin and activity are added up.</p></div>",
            unsafe_allow_html=True)
why("the model compares what happened before and after each moment. That only works if every input lines up on "
    "the same clock, whatever device it came from.")
if next_button(2, "clean the signal"):
    st.stop()

# ---------- 3 ----------
chapter(3, "Cleaning out sensor jitter", ["NumPy"])
z = history[(history.index >= t0 - pd.Timedelta(hours=30)) & (history.index < t0 - pd.Timedelta(hours=22))]
fig = go.Figure()
line(fig, z.index, z["glucose_raw"], "what the sensor said", P["tx3"], 3)
line(fig, z.index, z["glucose"], "cleaned", P["blue"], 1.8)
show(fig, 280, yaxis_title="mg/dL")
turn = lambda y: int((np.diff(np.sign(np.diff(y.dropna().to_numpy()))) != 0).sum())
st.markdown(f"<div class='story'><p>Glucose sensors wobble: tiny ups and downs that are noise, not biology. Any wiggle "
            f"smaller than 1 % of the current level is flattened. In Sam's week that removes "
            f"{1 - turn(history['glucose']) / max(turn(history['glucose_raw']), 1):.0%} of the turning points; across type 1 "
            "sensors (which wobble more) about 26 %.</p><p>The model learns from the cleaned line. Every score on this "
            "site is against the raw sensor, so cleaning can't flatter the results.</p></div>", unsafe_allow_html=True)
why("a fake dip-and-rise looks like a real reaction. Learning from noise would teach the model patterns that "
    "aren't there.")
if next_button(3, "who is Sam?"):
    st.stop()

# ---------- 4 ----------
chapter(4, "Who is Sam? Two kinds of body, two models", ["scikit-learn logistic regression"])
from classifier import classify  # noqa: E402
from engine import load_models, code_version  # noqa: E402
_, clf = load_models(code_version())
guess = classify({}, history[history.index < t0], clf)
probs = guess["probabilities"] or {}
c1, c2 = st.columns([1, 1.3], gap="large")
with c1:
    fig = go.Figure(go.Bar(x=[probs.get(g, 0) for g in TYPE_NAME], y=[TYPE_NAME[g] for g in TYPE_NAME], orientation="h",
                           marker_color=[P["green"], P["yellow"], P["orange"], P["magenta"]], hovertemplate="%{x:.0%}"))
    show(fig, 230, xaxis_tickformat=".0%", xaxis_range=[0, 1], showlegend=False)
with c2:
    st.markdown("<div class='story'><p>If you tell us your type (or your A1c, or we see mealtime insulin), we use that. If "
                "not, a small classifier guesses it from a week of glucose alone: average level, swings, time above "
                "140 and 180, time below 70, overnight level, typical rise after meals, activity.</p>"
                f"<p>From glucose alone it says <b>{TYPE_NAME.get(guess['group'], guess['group'])}</b> "
                f"({guess['reason']}). Sam's record says type 2. Close neighbours get mixed up (67 % right when "
                "tested on people it hadn't seen), but it found every type 1 person, which is the split that matters.</p>"
                "</div>", unsafe_allow_html=True)
avg = explain.get("average_response")
if avg:
    fig = go.Figure()
    mins = np.arange(0, 185, 5)
    for g, col in zip(("healthy", "prediabetes", "t2d", "t1d"), ("green", "yellow", "orange", "magenta")):
        line(fig, mins, avg[g]["mean"], f"{TYPE_NAME[g]} ({avg[g]['meals']} meals)", P[col], 2.6)
    show(fig, 320, xaxis_title="minutes after a 30 to 80 g carb meal", yaxis_title="glucose change (mg/dL)")
st.markdown("<div class='story'><p>Why route at all? Look at the average meal above for each type, from our datasets. "
            "With a working pancreas (healthy, prediabetes, type 2), the body releases its own insulin in step with the "
            "food: glucose rises, peaks, and comes back by itself. In type 1, the body makes no insulin at all. What "
            "happens depends on the insulin the person injects, when, and how much. Same carbs, a completely different "
            "system.</p><p>So there are two backends: a <b>metabolic</b> one (trained on 45 people from CGMacros: healthy, "
            "prediabetes, type 2) where 'own insulin' is something the model learns from the carbs, and a <b>type 1</b> "
            "one (trained on 334 people from five type 1 studies) where every insulin dose is an input.</p></div>",
            unsafe_allow_html=True)
why("one model for everyone would average two fundamentally different systems and be wrong for both.")
if next_button(4, "read the plate"):
    st.stop()

# ---------- 5 ----------
chapter(5, "Reading the plate", ["CLIP ViT-L/14 (OpenAI, MIT licence)", "USDA FNDDS (public domain)", "PyTorch", "Hugging Face"])
ph = explain.get("photo")
c1, c2 = st.columns([1, 1.3], gap="large")
c1.image(str(DEMO / "meal_photo.jpg"), width="stretch")
with c2:
    if ph:
        rows = "".join(f"<div>{s:.0%} &nbsp;{f} <span class='small'>· usually {p}, {g:.0f} g</span></div>"
                       for f, p, g, s in ph["items"][:5])
        n = ph["nutrition"]
        st.markdown(f"<div class='small'>What the model sees:</div><div style='line-height:1.7;margin:.3rem 0 .8rem'>{rows}</div>"
                    f"<div>Estimate: <span class='c-yellow'>{n['carbs']:.0f} g carbs</span> · {n['protein']:.0f} g protein · "
                    f"{n['fat']:.0f} g fat · {n['calories']:.0f} kcal <span class='small'>(Sam logged {meal['carbs']:.0f} g carbs)"
                    "</span></div>", unsafe_allow_html=True)
st.markdown("<div class='story'><p>Three steps, all open:</p><ol>"
            "<li><b>Recognise.</b> CLIP is a neural network trained on 400 million image and caption pairs, so it can "
            "tell how well a photo matches any text. We ask it how well the photo matches each of about 5,200 foods "
            "in the USDA food database (\"a photo of burrito bowl, a type of food\"). No extra training needed.</li>"
            "<li><b>Look up.</b> For every food, USDA FNDDS lists nutrients per 100 g and the amount people typically eat "
            "when they don't say how much (from the US national diet survey). The estimate is the confidence-weighted "
            "mix of the top matches at their typical portion.</li>"
            "<li><b>Remember.</b> People eat the same meals again and again. If the photo looks very much like a meal "
            "logged before, that meal's real numbers count for more.</li></ol>"
            "<p>Then you set the portion (0 to 3 times the usual) and fix anything that looks off.</p></div>",
            unsafe_allow_html=True)
pscores = ROOT / "results" / "photo2nutrition_scores.csv"
if pscores.exists():
    sc = pd.read_csv(pscores, index_col=0)
    st.dataframe(sc.rename(columns=lambda c: f"{c} (g)" if c != "calories" else "kcal").style.format("{:.1f}"),
                 width="stretch")
    st.markdown("<p class='small'>Average miss per meal on photos from day 8 on (never used for tuning). A photo can't "
                "show portion size or what's hidden inside, so this is a helper, not a replacement for reading the "
                "label.</p>", unsafe_allow_html=True)
why("carbs drive the spike, and most people don't know the carbs in a home-cooked plate. A rough, honest starting "
    "point you can correct beats a blank box.")
if next_button(5, "add the context"):
    st.stop()

# ---------- 6 ----------
chapter(6, "Everything else we know about this moment", ["pandas"])
from context import meal_table  # noqa: E402
s_meal = plan_series(history, t0, t0 + AFTER, BACKEND, [meal])
row = meal_table(s_meal[s_meal.index <= t0]).iloc[-1]
kv = [("Meal of the day", ["first", "second", "third or later"][int(row["slot"]) - 1]),
      ("Carbs already eaten today", f"{row['carbs_today']:.0f} g"),
      ("Time since the last meal", f"{row['mins_since_meal'] / 60:.1f} h"),
      ("Glucose right now vs Sam's usual", f"{row['pre_vs_normal']:+.0f} mg/dL"),
      ("Sam's usual level (last 3 days)", f"{row['usual_level']:.0f} mg/dL"),
      ("How much Sam's glucose swings", f"± {row['swing']:.0f} mg/dL"),
      ("Sam's own past rise per 10 g carbs", f"{row['past_rise_per_10g']:.1f} mg/dL" if pd.notna(row["past_rise_per_10g"]) else "not yet known")]
st.markdown("<div class='kv'>" + "".join(f"<div class='k'>{k}</div><div>{v}</div>" for k, v in kv) + "</div>",
            unsafe_allow_html=True)
st.markdown("<div class='story'><p>The same bowl lands differently depending on the day so far. These values become "
            "extra inputs: carbs split by meal of the day, and carbs scaled by each value. So the model can learn "
            "things like \"a second meal soon after the first spikes less\", \"exercise before a meal blunts it\", or "
            "\"Sam spikes about this much per 10 g\". Everything here comes from <b>before</b> the meal.</p></div>",
            unsafe_allow_html=True)
why("a carb count alone explains only part of a spike. The rest is context, and context is cheap to compute from "
    "data you already have.")
if next_button(6, "turn events into curves"):
    st.stop()

# ---------- 7 ----------
chapter(7, "Turning food and insulin into curves over time", ["SciPy (fftconvolve)", "OpenAPS oref0 insulin curve (MIT)"])
from exogenous import bump, insulin_used, ramp  # noqa: E402
t = np.arange(0, 361, 5)
fig = go.Figure()
for p_, col in ((20, "yellow"), (60, "orange"), (150, "red")):
    line(fig, t, bump(p_)(t), f"food, peaks at {p_} min", P[col], 2.2)
line(fig, t, ramp(180)(t), "lasting shift", P["tx2"], 2, "dot")
act = np.gradient(insulin_used()(t), t)
line(fig, t, act / act.max(), "insulin activity (OpenAPS)", P["blue"], 2.6)
show(fig, 300, xaxis_title="minutes after eating or injecting", yaxis_title="effect (scaled to 1)")
st.markdown("<div class='story'><p>Food doesn't hit the blood at once: it digests over minutes to hours. Each input is "
            "smeared through a few fixed shapes like the ones above (fast, medium and slow bumps, a lasting shift). "
            "The model then learns <i>how much of each shape</i> Sam has, which gives Sam's own response curve.</p>"
            "<p>Insulin uses the OpenAPS insulin curve, the same open-source math insulin pumps use for \"insulin on "
            "board\": rapid insulin peaks around 75 minutes and is used up after about 6 hours. For type 2 and healthy "
            "people the model gives carbs a faster copy of this curve as \"own insulin\" and learns its strength per "
            "person. Exercise and quiet stretches (overnight, mid-afternoon) are treated the same way.</p></div>",
            unsafe_allow_html=True)
why("a fixed library of realistic shapes keeps the model linear and fast, and means even a few days of data can "
    "pin down a person's own curve.")
if next_button(7, "the model itself"):
    st.stop()

# ---------- 8 ----------
chapter(8, "The model: from now, plus every push and pull", ["NumPy least squares"])
person = sam()
model, coef = person["model"], person["post"][0]
anchor = int(np.flatnonzero(s_meal.index < t0)[-1])
mean_parts, _ = model.components(s_meal[COLUMNS], anchor, person["post"], person["base"])
mean_parts = mean_parts.loc[mean_parts.index <= 180]
fig = go.Figure()
cols = [P[c] for c in ("yellow", "blue", "cyan", "purple", "orange", "green", "red", "magenta")]
for (name, v), col in zip(mean_parts.loc[:, mean_parts.abs().max() > 1].items(), cols):
    line(fig, mean_parts.index, v, name, col, 2.2)
line(fig, mean_parts.index, mean_parts.sum(axis=1), "total change", P["tx"], 3, "dash")
show(fig, 340, xaxis_title="minutes after lunch", yaxis_title="contribution to glucose change (mg/dL)")
st.markdown("<div class='story'><p>Starting from Sam's glucose right now, the prediction is a sum of named pushes and "
            "pulls: keep going the way it was heading (recent trend), drift back to Sam's normal level, the food curves "
            "from the last chapter, Sam's own daily rhythm, activity. Each has a weight, and the weights are fit by least "
            "squares on thousands of moments: \"from here, what happened 5, 10, ... 240 minutes later?\". This is called "
            "an ARX model (autoregressive with exogenous inputs).</p><p>There is a second model, the <b>overlay</b>: a "
            "baseline plus three rising and three falling S-curves (sigmoids) per event, whose sizes and timing depend "
            "on the meal and its context. The site uses whichever "
            "scored better for that type of person.</p></div>", unsafe_allow_html=True)
why("deep networks need far more data per person than a week gives. A "
    "linear model with good inputs learns from little data, can't do anything wild, and every part can be explained "
    "(as above).")
if next_button(8, "learn Sam, not the average"):
    st.stop()

# ---------- 9 ----------
chapter(9, "Learning Sam, not the average person", ["ridge regression", "Bayesian shrinkage"])
pop = model.population
grp = model.prior(person["fit_group"], exclude=about["dataset_id"] if about["dataset_id"] in getattr(model, "person_stats", {}) else None)
fig = go.Figure()
known = history["glucose_raw"][(history.index >= t0 - pd.Timedelta(hours=2)) & (history.index < t0)]
line(fig, known.index, known, "Sam before lunch", P["red"], 2.4)
rows = []
for name, c, col in (("everyone", pop, "tx3"), (f"people with {TYPE_NAME[person['group']]}", grp, "orange"), ("Sam", coef, "magenta")):
    p_, _ = predict_window(model, s_meal[COLUMNS], (c, np.zeros((len(c), len(c))), 0.0), t0, t0 + AFTER, person["base"])
    line(fig, p_.index, p_, f"learned from {name}", P[col], 2.6 if name == "Sam" else 2, None if name == "Sam" else "dot")
    e = (p_ - truth).dropna()
    rows.append((name, float(np.sqrt(np.mean(e ** 2)))))
line(fig, truth.index, truth, "what really happened", P["green"], 2.6)
show(fig, 360, yaxis_title="mg/dL")
st.markdown("<div class='story'><p>The weights are learned in three layers. First from <b>everyone</b> in the training "
            "data. Then nudged toward <b>people of Sam's type</b>. Then toward <b>Sam's own week</b>. Each layer only "
            "moves as far as its data justifies: with a day of data Sam's curve stays close to the group, with a week "
            "it becomes Sam's. (Statistically: each layer is the prior for the next, ridge regression toward it.) How "
            "hard each layer pulls was tuned on held-out days.</p>"
            f"<p>For this lunch, the typical miss (RMSE) was {rows[0][1]:.0f} mg/dL learning from everyone, "
            f"{rows[1][1]:.0f} from Sam's type and <b>{rows[2][1]:.0f} from Sam</b>. Sam was taken out of the "
            "first two layers, so Sam is never-seen.</p></div>", unsafe_allow_html=True)
why("people really are different: the same bowl spikes one person by 30 and another by 90. Starting from the crowd "
    "and moving toward the person gives a sensible answer on day one and a personal one after a week.")
if next_button(9, "how sure are we?"):
    st.stop()

# ---------- 10 ----------
chapter(10, "How sure are we?", ["posterior covariance", "quantile calibration"])
pred = predict_after_meal(person, history, meal)
fig = go.Figure()
line(fig, known.index, known, "Sam before lunch", P["red"], 2.4)
fig.add_scatter(x=pred.index, y=pred["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=pred.index, y=pred["lo"], fill="tonexty", line=dict(width=0), fillcolor=rgba(P["magenta"], 0.16),
                name="80 % band", hoverinfo="skip")
line(fig, pred.index, pred["pred"], "prediction", P["magenta"], 2.6, "dash")
line(fig, truth.index, truth, "what really happened", P["green"], 2.6)
show(fig, 340, yaxis_title="mg/dL")
track = track_record(person, history)
cov_txt = ""
if scores is not None:
    sc = scores[(scores["variant"] == "person") & (scores["kind"] == "regressor")].set_index("backend")
    cov_txt = " ".join(f"For {'people without type 1' if b == 'metabolic' else 'type 1'}, the truth landed inside "
                       f"{np.nanmean([sc.loc[b, c] for c in ('cov30', 'cov60', 'cov90', 'cov120')]):.0f} % of the time."
                       for b in sc.index)
st.markdown("<div class='story'><p>Two sources of doubt go into the band. How unsure we are about Sam's weights (less "
            "data, more doubt, worked out exactly from the least squares fit), plus how much glucose wanders that no "
            "input explains. Over a whole day each new step builds on the last guess, so the band widens.</p>"
            "<p>Then the band is checked against reality on days the model didn't learn from, and set so that it holds "
            "the truth 80 % of the time. Errors after meals are lopsided (big spikes run higher than predicted more "
            f"often than lower), so the band is too: it reaches further up than down. {cov_txt}</p>"
            + (f"<p>And for Sam specifically: re-predicting Sam's last {track['meals']} meals from the days before them, "
               f"the average miss was {track['mae']:.0f} mg/dL. That number is shown next to every prediction.</p>"
               if track else "") + "</div>", unsafe_allow_html=True)
why("a prediction without an honest error bar is a guess pretending to be a fact. If we say 80 %, it should be 80 %.")
if next_button(10, "did it work?"):
    st.stop()

# ---------- 11 ----------
chapter(11, "Did it work?")
if scores is not None:
    sc = scores[scores["variant"].isin(["flat", "person"])].copy()
    sc["who"] = sc["backend"].map({"metabolic": "healthy, prediabetes, type 2", "t1d": "type 1"})
    sc["what"] = np.where(sc["variant"] == "flat", "no model (glucose stays flat)", sc["kind"])
    tab = sc.drop_duplicates(["backend", "what"])[["who", "what", "rmse_err30", "rmse_err60", "rmse_err90", "rmse_err120", "rmse_all", "meals"]]
    st.dataframe(tab.rename(columns={"who": "people", "what": "model", "rmse_err30": "30 min", "rmse_err60": "60 min",
                                     "rmse_err90": "90 min", "rmse_err120": "120 min", "rmse_all": "all"}),
                 hide_index=True, width="stretch")
st.markdown("<div class='story'><p>Typical miss (RMSE, mg/dL) 30 to 120 minutes after every meal, on people the model "
            "never trained on: it learns their first 7 days, then predicts every meal after that, scored against the raw "
            "sensor. The flat line (\"glucose stays where it is\") is the bar to beat.</p>"
            "<p>Type 1 is much harder: insulin timing, unlogged snacks and pumps that adjust insulin on their own all "
            "move glucose in ways no meal log shows.</p></div>", unsafe_allow_html=True)

st.markdown("<div class='chapter'>The building blocks</div><div class='story'><ul>"
            "<li><b>Data:</b> CGMacros (45 people with glucose, meals, photos, activity; PhysioNet) and MetaboNet type 1 "
            "studies (Loop, PEDAP, AZT1D, HUPA-UCM, OhioT1DM), USDA FNDDS 2021-2023 food database.</li>"
            "<li><b>Models:</b> our own ARX and overlay models in NumPy and SciPy, the OpenAPS oref0 insulin curve, a "
            "scikit-learn logistic regression to pick the type, CLIP ViT-L/14 through PyTorch and Hugging Face "
            "transformers for photos.</li>"
            "<li><b>Site:</b> Streamlit, Plotly, Flexoki colours, the Libron typeface, Gray's Anatomy plates (1918, "
            "public domain).</li>"
            "<li>No proprietary model or paid API anywhere.</li></ul>"
            "<p class='small'>Glucklich predicts; it does not give medical advice.</p></div>", unsafe_allow_html=True)
