"""Home: what this is, in one picture, and a three-step demo.

Loads in stages so the page appears at once: the hero, art and facts need only the light style module; the demo
has a placeholder until the plotting code is loaded. Nothing is learned or predicted until Predict! is pressed
(then it is kept for every visitor, so the next time is instant). The photo model loads only when asked.
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from style import TYPE_COLOR, TYPE_NAME, brand, css, photo_file

ROOT = Path(__file__).resolve().parent.parent
DEMO, ART = ROOT / "samples" / "home_demo", ROOT / "static" / "art"
css()
st.markdown("""<style>
  .kicker {font-size: .95rem; letter-spacing: .16em; text-transform: uppercase; color: var(--magenta); margin-top: 1.5rem;}
  .hero {font-size: 3.4rem; line-height: 1.04; margin: .3rem 0 1rem; font-weight: 700;}
  .hero-sub {font-size: 1.35rem; line-height: 1.45; color: var(--tx); max-width: 34rem;}
  .credit {font-size: .8rem; color: var(--tx2); font-style: italic; text-align: center;}
  .pasta {font-size: 1.9rem; line-height: 1.3; text-align: center; margin: 3.2rem auto 0; max-width: 46rem;}
  .pasta em {color: var(--magenta);}
  .facts {display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 2rem; margin: 3.2rem 0 2.4rem;}
  .facts .num {font-size: 3rem; line-height: 1; font-weight: 700;}
  .facts p {margin: .5rem 0 0; color: var(--tx2); font-size: 1.02rem; line-height: 1.45;}
  .accurate {font-size: 2rem; margin: 1rem 0 0;}
  .meal-card {border: 1px solid var(--ui); border-radius: 10px; padding: 1rem 1.2rem;}
  .plate {position: relative; width: 100%; margin: auto;}
  .plate > div {position: absolute; inset: 0; -webkit-mask-size: contain; mask-size: contain;
                -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat; -webkit-mask-position: center; mask-position: center;}
  .waiting {color: var(--tx2); font-size: 1.1rem; padding: 2rem 0; animation: pulse 1.4s ease-in-out infinite;}
  @keyframes pulse {50% {opacity: .45;}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def mask(path):
    """A plate layer as an inline data URI (no extra request) and the plate's size."""
    return "data:image/webp;base64," + base64.b64encode(Path(path).read_bytes()).decode(), Image.open(path).size


def plate(name, max_width="100%"):
    """A Gray's Anatomy plate without its paper. Its ink and arteries are masks coloured by the theme's
    --art_ink / --art_red, so one inline copy serves both themes and switches instantly."""
    layers = [(ART / f"{name}_{part}.webp", var) for part, var in (("ink", "art_ink"), ("red", "art_red"))]
    html, size = "", (1, 1)
    for path, var in layers:
        if path.exists():
            uri, size = mask(str(path))
            html += f"<div style='background:var(--{var});-webkit-mask-image:url({uri});mask-image:url({uri})'></div>"
    return f"<div class='plate' style='max-width:{max_width};aspect-ratio:{size[0]}/{size[1]}'>{html}</div>"


def glucose_art(series):
    """One real day of glucose, drawn as a line (no smoothing, no styling of the data itself)."""
    g = series["glucose"].dropna()
    x = (g.index - g.index[0]).total_seconds() / 86400 * 1000
    lo_, hi_ = g.min() - 10, g.max() + 10
    y = 150 - (g - lo_) / (hi_ - lo_) * 130
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    dots = "".join(f"<circle cx='{(t - g.index[0]).total_seconds() / 86400 * 1000:.1f}' cy='158' r='4' style='fill:var(--yellow)'/>"
                   for t in series.index[series["carbs"] > 0] if g.index[0] <= t <= g.index[-1])
    return f"""<svg viewBox="0 0 1000 170" preserveAspectRatio="none" style="width:100%;height:150px;display:block">
      <defs><linearGradient id="gl" x1="0" x2="1"><stop offset="0" style="stop-color:var(--blue)"/><stop offset=".35" style="stop-color:var(--cyan)"/>
        <stop offset=".6" style="stop-color:var(--orange)"/><stop offset="1" style="stop-color:var(--magenta)"/></linearGradient></defs>
      <polyline points="{pts}" fill="none" stroke="url(#gl)" stroke-width="2.6" stroke-linejoin="round"/>{dots}</svg>"""


# ---------- Hero ----------
left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
with left:
    brand()
    st.markdown("<div class='kicker'>glucose, before it happens</div>"
                "<div class='hero'>What will that meal do to your blood sugar?</div>"
                "<div class='hero-sub'>Show us what's on your plate and we'll sketch your next three hours. "
                "Spikes, dips, the lot!</div>", unsafe_allow_html=True)
with right:
    st.markdown(plate("gray1105_islets", "440px") +
                "<div class='credit'>The islets in the pancreas, where insulin is made. Gray's Anatomy, 1918.</div>",
                unsafe_allow_html=True)

day = pd.read_csv(ART / "glucose_day.csv", parse_dates=["time"]).set_index("time")
st.markdown(glucose_art(day) + "<div class='credit'>One real day of someone's glucose, as the sensor saw it. "
            "The yellow dots are meals.</div>"
            "<div class='pasta'><em>Same bowl of pasta, different person, totally different spike.</em> "
            "So we learn <span class='c-red'>you</span> first.</div>", unsafe_allow_html=True)

st.markdown("""<div class='facts'>
  <div><div class='num c-red'>1 in 9</div><p>adults in the world live with diabetes, about 589 million people. Around
     4 in 10 of them don't know it yet. <span class='small'>(IDF Diabetes Atlas, 2025)</span></p></div>
  <div><div class='num c-yellow'>Every meal</div><p>millions of people who take insulin have to guess how high their
     sugar will go, then dose for it. Too little and it climbs, too much and it crashes. About 9 million people live
     with type 1 alone. <span class='small'>(T1D Index)</span></p></div>
  <div><div class='num c-magenta'>180</div><p>extra health decisions a day, roughly, for someone with type 1: what
     to eat, how much insulin, when to check, whether that walk is safe. Most of them are guesses about what their
     glucose will do next. <span class='small'>(Stanford Medicine, 2014)</span></p></div>
</div>""", unsafe_allow_html=True)

c1, c2 = st.columns([1, 1.2], gap="large", vertical_alignment="center")
with c1:
    st.markdown(plate("gray1100_pancreas") + "<div class='credit'>The pancreas. Gray's Anatomy, 1918.</div>",
                unsafe_allow_html=True)
with c2:
    st.markdown("<div class='hero-sub'>Your pancreas has been doing this math quietly your whole life. If it stopped "
                "(type 1) or got a bit lazy (type 2), you suddenly have to guess.<br><br>"
                "<b>So let's guess better.</b></div>", unsafe_allow_html=True)

# ---------- The demo frame (filled in below, once the models are loaded) ----------
st.markdown("<div class='step-no' style='margin-top:4rem'>Predict in three easy steps</div>", unsafe_allow_html=True)
demo_box = st.container()
waiting = demo_box.empty()
waiting.markdown("<div class='waiting'>Getting our demo people ready...</div>", unsafe_allow_html=True)

st.markdown("<div class='step-no' style='margin-top:3.5rem'>Your turn</div>"
            "<div class='step-title'>Try it yourself!</div>"
            "<p class='muted'>Drop in your own glucose export and your next meal, or plan a whole day. Sample people are "
            "waiting on those pages if you just want to play.</p>", unsafe_allow_html=True)
b1, b2, _ = st.columns([1, 1, 2])
if b1.button("Predict your own", type="primary", key="go_try", width="stretch"):
    st.switch_page("views/meal.py")
if b2.button("Predict a whole day", key="go_day", width="stretch"):
    st.switch_page("views/full_day.py")

# ---------- The demo ----------
from engine import (_shared_store, code_version, history_plot, learn, meal_plot, photo_model,  # noqa: E402
                    predict_after_meal, read, rise_text, score, track_record,
                    accuracy_note, commentary_html, image_picker, meal_commentary,
                    nutrition_line, option_picker)

PEOPLE = {p.name: p for p in sorted(DEMO.iterdir(), key=lambda p: p.name != "Maya") if p.is_dir()}
ABOUT = {n: pd.read_csv(f / "about_me.csv").iloc[0] for n, f in PEOPLE.items()}
st.session_state.setdefault("home_person", next(iter(PEOPLE)))
SHARED = _shared_store(code_version())


def demo_person(n):
    """Learn a demo person (only when Predict! is pressed), once for every visitor. They stay out of the priors,
    so they are never-seen. Also their track record: how far off we were on their own last two days of meals."""
    ab = ABOUT[n]
    history = read(PEOPLE[n] / "history.csv")
    person = learn(history, {"group": ab["diabetes_type"], **({"a1c": ab["a1c"]} if pd.notna(ab.get("a1c")) else {})},
                   f"demo-{n}", exclude=ab.get("dataset_id"), shared=True)
    saved = PEOPLE[n] / "track_record.npz"                 # replayed ahead of time (models/demo_tracks.py)
    if "_track" not in person and saved.exists():
        t = np.load(saved)
        person["_track"] = {"resid": t["resid"], "mae": float(t["mae"]), "rmse": float(t["rmse"]), "meals": int(t["meals"])}
    track_record(person, history)
    return person


with demo_box:
    waiting.empty()
    name = st.session_state.home_person
    t1 = ABOUT[name]["diabetes_type"] == "t1d"
    folder, about = PEOPLE[name], ABOUT[name]
    history = read(folder / "history.csv")

    st.markdown("<div class='step-no'>Step 1</div><div class='step-title'>Tell us about yourself</div>"
                "<p class='muted'>Pick someone. We already have their past week.</p>", unsafe_allow_html=True)
    cols = st.columns(len(PEOPLE) + 2)
    for col, (n, ab) in zip(cols, ABOUT.items()):
        if col.button(f"{n}", key=f"pick_{n}", type="primary" if name == n else "secondary", width="stretch"):
            st.session_state.home_person = n
            st.rerun()
        col.markdown(f"<div class='small c-{TYPE_COLOR[ab['diabetes_type']]}' style='text-align:center'>"
                     f"{TYPE_NAME[ab['diabetes_type']]}</div>", unsafe_allow_html=True)

    meal_row = pd.read_csv(folder / "meal.csv").iloc[0]
    doses = pd.read_csv(folder / "doses.csv", parse_dates=["time"]) if (folder / "doses.csv").exists() else pd.DataFrame(columns=["time", "units"])
    days = (history.index[-1] - history.index[0]).total_seconds() / 86400
    s1, s2 = st.columns([1, 2.2], gap="large")
    s1.markdown(f"<div style='font-size:1.7rem;margin-top:1rem'>{name}</div>"
                f"<div class='c-{TYPE_COLOR[about['diabetes_type']]}'>{TYPE_NAME[about['diabetes_type']]}</div>"
                f"<p class='muted' style='margin-top:.8rem'>{days:.0f} days of glucose, {int((history['carbs'] > 0).sum())} meals"
                f"{', every insulin dose' if t1 else ''}. Usually around {history['glucose'].median():.0f} mg/dL, give or take {history['glucose'].std():.0f} (standard deviation).</p>",
                unsafe_allow_html=True)
    with s2:
        history_plot(history, days=2, height=260)

    st.markdown("<div class='step-no'>Step 2</div><div class='step-title'>What are you eating?</div>", unsafe_allow_html=True)
    key = lambda f: f"home_{name}_{f}"
    vals = {f: float(meal_row.get(f, 0) or 0) for f in ["carbs", "protein", "fat", "fiber", "calories"]}
    if t1:
        vals["insulin"] = float(meal_row.get("insulin", 0) or 0)
    meal_time = pd.Timestamp(meal_row["time"])
    m1, m2 = st.columns([1, 1.3], gap="large", vertical_alignment="center")
    with m1:
        photo = folder / "meal_photo.jpg"
        if photo.exists():
            st.image(str(photo), width="stretch")
        else:
            st.markdown(f"<div class='meal-card' style='font-size:1.15rem'>No photo for this one, {name} logged it by hand.</div>",
                        unsafe_allow_html=True)
    with m2:
        if photo.exists():
            if st.button("Read the nutrition from the photo", key=key("photo_btn"), width="stretch"):
                with st.spinner("Looking at the plate..."):
                    saved = folder / "photo_analysis.json"         # read when the demo files were built
                    res = json.loads(saved.read_text()) if saved.exists() else None
                    if res is None or "options" not in res:
                        pm = photo_model()
                        res = pm.analyse(str(photo))
                        res["options"] = [{"name": pm.food(i)["name"], "nutrition": pm.plate([(i, 1.0)], res["memory"], res["memory_weight"])}
                                          for i, _ in res["candidates"]]
                    st.session_state[key("photo")] = res
                    st.session_state[key("pick")] = ("option", 0)
            res = st.session_state.get(key("photo"))
            if res:
                kind_, i_ = st.session_state.get(key("pick"), ("option", 0))
                near = [dict(n, photo=f) for n in res.get("neighbours", []) if (f := photo_file(n.get("photo")))][:3]
                chosen_n = res["options"][i_]["nutrition"] if kind_ == "option" else near[i_]
                vals.update({f: round(float(chosen_n[f]), 1) for f in ("carbs", "protein", "fat", "fiber", "calories")})
                st.markdown("<p class='small' style='margin:.3rem 0 .2rem'>What CLIP sees (USDA nutrients, usual portion):</p>",
                            unsafe_allow_html=True)
                c = option_picker([(o["name"], nutrition_line(o["nutrition"]), kind_ == "option" and k == i_)
                                   for k, o in enumerate(res["options"])], key=key("opt_pick"))
                if c is not None:
                    st.session_state[key("pick")] = ("option", c)
                    st.rerun()
                if near:
                    st.markdown("<p class='small' style='margin:.4rem 0 .2rem'>Logged meals that look the same:</p>",
                                unsafe_allow_html=True)
                    own = lambda n: f"cgmacros-{n['subject']:03d}" == str(about.get("dataset_id", ""))
                    c = image_picker([(n["photo"], nutrition_line(n) + (f" · {name}'s own" if own(n) else ""),
                                       kind_ == "near" and k == i_) for k, n in enumerate(near)], key=key("near_pick"))
                    if c is not None:
                        st.session_state[key("pick")] = ("near", c)
                        st.rerun()
        nutr = [("yellow", vals["carbs"], "g carbs"), ("red", vals["protein"], "g protein"), ("orange", vals["fat"], "g fat"),
                ("green", vals["fiber"], "g fiber"), ("purple", vals["calories"], "kcal")]
        shown = [f"<span class='c-{c}'>{v:.0f} {u}</span>" for c, v, u in nutr if v > 0 or u == "g carbs"]
        st.markdown(f"<div style='font-size:1.5rem'>{meal_row['meal']} at {meal_time:%H:%M}</div>"
                    f"<div style='font-size:1.25rem;line-height:1.9'>{' · '.join(shown)}</div>", unsafe_allow_html=True)
        if t1:
            plan = [f"{vals['insulin']:.1f} U with the meal"] if vals["insulin"] > 0 else []
            plan += [f"{u:.1f} U at {t:%H:%M}" for t, u in zip(doses["time"], doses["units"])]
            st.markdown(f"<div class='c-orange' style='font-size:1.1rem'>Insulin plan: {', '.join(plan) or 'none'}</div>",
                        unsafe_allow_html=True)
        go_ = st.button("Predict!", type="primary", width="stretch", key=key("predict"))

    meal = {"time": meal_time, "name": meal_row["meal"], **vals, "doses": list(zip(doses["time"], doses["units"]))}
    sig = str(sorted(meal.items()))
    if go_:
        preds = SHARED.setdefault("_demo_preds", {})
        if (name, sig) not in preds:
            with st.spinner(f"Learning how {name} responds to food, then drawing the next three hours..."):
                preds[(name, sig)] = predict_after_meal(demo_person(name), history, meal)
        st.session_state[key("pred")] = (sig, preds[(name, sig)])
        st.session_state[key("truth")] = False

    if key("pred") in st.session_state:
        psig, pred = st.session_state[key("pred")]
        truth_df = pd.read_csv(folder / "truth.csv", parse_dates=["time"]).set_index("time")["glucose"]
        show_truth = st.session_state.get(key("truth"), False)
        meal_plot(history, meal, pred, truth_df if show_truth else None)
        st.markdown(f"<div class='accurate'>{rise_text(pred, history['glucose_raw'].dropna().iloc[-1], history['glucose'].median())}</div>"
                    + commentary_html(meal_commentary(pred, history, meal)), unsafe_allow_html=True)
        track = demo_person(name).get("_track")
        st.markdown(accuracy_note(pred, track, name),
                    unsafe_allow_html=True)
        if st.button("Hide what really happened" if show_truth else "See what really happened", key=key("truth_btn")):
            st.session_state[key("truth")] = not show_truth
            st.rerun()
        if show_truth:
            s = score(pred, truth_df)
            if s:
                k1, k2, _ = st.columns(3)
                k1.markdown(f"<div class='small'>This meal: typical miss (RMSE)</div><div class='big-number c-green'>{s['rmse']:.0f}</div>"
                            "<div class='small'>mg/dL from the predicted line</div>", unsafe_allow_html=True)
                k2.markdown(f"<div class='small'>Truth inside the band</div><div class='big-number c-green'>{s['inside']:.0%}</div>"
                            f"<div class='small'>of {s['n']} readings</div>", unsafe_allow_html=True)
