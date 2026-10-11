"""Shared by the pages: loading models, reading files, learning a person, predicting after a meal, scoring, plots."""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
for p_ in (ROOT, ROOT / "models"):
    if str(p_) not in sys.path:
        sys.path.insert(0, str(p_))
MODEL_FILES = ["parse", "clean", "common", "context", "events", "exogenous", "regressor", "overlay", "classifier", "decompose", "plan"]


def code_version():
    """Changes whenever the weights or model code change."""
    files = [ROOT / "models" / "weights" / "models.pkl", *(ROOT / "models" / f"{name}.py" for name in MODEL_FILES)]
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
from common import COLUMNS, WEIGHTS, from_csv, baseline, predict_window, solve  # noqa: E402
from classifier import classify, load_classifier  # noqa: E402
from plan import INTENSITY, NUTRIENTS, plan_series  # noqa: E402
from parse import read_table  # noqa: E402,F401

from style import DARK, FONT, LIGHT, TYPE_COLOR, TYPE_NAME, Z80, brand, css, palette, theme  # noqa: E402,F401

AFTER = pd.Timedelta(hours=3)


# ---------- Loading ----------

@st.cache_resource
def load_models(version):
    with open(WEIGHTS / "models.pkl", "rb") as f:
        return pickle.load(f), load_classifier()


IN_BROWSER = sys.platform == "emscripten"           # the site running in the browser (stlite on GitHub Pages)


@st.cache_resource
def _photo_model(version):
    """The photo module, reloaded when its file changes. Loads only the fast image model (OpenVINO when available)."""
    import importlib
    import photo2nutrition
    photo2nutrition = importlib.reload(photo2nutrition)
    if not IN_BROWSER and photo2nutrition.fast_vision() is None:    # the browser gets its vectors from browser_clip
        photo2nutrition.vision()
    photo2nutrition.foods_and_text()
    return photo2nutrition


CLIP_JS = r"""
export default async function(component) {
  const { data, setTriggerValue, parentElement } = component;
  const el = parentElement.querySelector('.clip');
  if (el.dataset.key === data.key) return;                 // already working on this photo
  el.dataset.key = data.key;
  const say = t => { el.textContent = t; };
  try {
    say('Loading the photo model...');
    const T = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js');
    const seen = {};
    window.__glkClip = window.__glkClip || Promise.all([
      T.AutoProcessor.from_pretrained('Xenova/clip-vit-large-patch14'),
      T.CLIPVisionModelWithProjection.from_pretrained('Xenova/clip-vit-large-patch14', { dtype: 'q4', device: 'wasm',
        progress_callback: p => {
          if (p.status === 'progress' && p.total > 1e6) {
            seen[p.file] = [p.loaded, p.total];
            const [l, t] = Object.values(seen).reduce((a, b) => [a[0] + b[0], a[1] + b[1]], [0, 0]);
            say(`Downloading the photo model, once (${Math.round(t / 1e6)} MB): ${Math.round(100 * l / t)}%`);
          }
        } }),
    ]);
    const [proc, model] = await window.__glkClip;
    say('Looking at the photo...');
    const image = await T.RawImage.fromURL(data.image);
    const { image_embeds } = await model(await proc(image));
    const v = Array.from(image_embeds.data);
    const n = Math.hypot(...v);
    setTriggerValue('vector', v.map(x => x / n));
  } catch (e) {
    el.dataset.key = '';
    say('Could not read the photo here: ' + e.message);
  }
}
"""


def browser_clip(data, key):
    """In the browser: CLIP's image vector for a photo (bytes), computed by transformers.js (the same ViT-L/14, 4-bit).
    Returns the vector once it's ready, else None (the progress shows meanwhile and the page reruns when done)."""
    import base64
    done = st.session_state.setdefault("_clip_vectors", {})
    if key in done:
        return done[key]
    if "_clip" not in st.session_state:
        st.session_state["_clip"] = st.components.v2.component(
            "browser_clip", html="<div class='clip' style='font-family:Libron,Georgia,serif;opacity:.8'></div>", js=CLIP_JS)
    res = st.session_state["_clip"](key=f"clip_{key}", data={"key": key, "image": "data:image/jpeg;base64," +
                                    base64.b64encode(data).decode()}, on_vector_change=lambda: None)
    v = getattr(res, "vector", None)
    if v is not None:
        done[key] = v
    return v


def photo_model():
    return _photo_model((ROOT / "models" / "photo2nutrition.py").stat().st_mtime)


@st.cache_data(show_spinner=False, max_entries=64)
def _read_path(path, mtime):
    return from_csv(path)


def read(file):
    """A path (cached) or an uploaded file -> 5-min series with 'glucose' cleaned and 'glucose_raw'."""
    if file is None:
        return None
    if isinstance(file, Path):
        return _read_path(str(file), file.stat().st_mtime)
    file.seek(0)
    return from_csv(file)


DATA_TYPES = ["csv", "txt", "tsv", "xlsx", "xls", "json", "zip"]
PHOTO_TYPES = ["jpg", "jpeg", "png", "webp"]


def drop_box(label, types, key):
    """An uploader for files, a whole folder or a zip, in any mix. Dragging a folder in works either way; the switch
    makes Browse open a folder picker. Returns the files, with zips opened and folders flattened."""
    with st.container(key=key):
        folder = st.toggle("Browse for a folder", key=f"{key}_folder")
        got = st.file_uploader(label, type=types, accept_multiple_files="directory" if folder else True,
                               key=f"{key}_{'dir' if folder else 'files'}", label_visibility="collapsed")
    return unpack(got)


def unpack(files):
    """Uploaded files, with every .zip opened into its files (subfolders flattened, system files skipped)."""
    import io
    import zipfile
    out = []
    for f in files or []:
        name = f.name.replace("\\", "/").split("/")[-1]
        if name.startswith((".", "~$")) or "__MACOSX" in f.name:
            continue
        if not name.lower().endswith(".zip"):
            out.append(f)
            continue
        f.seek(0)
        with zipfile.ZipFile(f) as z:
            for n in z.namelist():
                inner = n.split("/")[-1]
                if not inner or inner.startswith((".", "~$")) or "__MACOSX" in n:
                    continue
                b = io.BytesIO(z.read(n))
                b.name = inner
                out.extend(unpack([b]) if inner.lower().endswith(".zip") else [b])
    return out


def read_readings(files, start=None, end=None):
    """Glucose readings from any pile of files (one export, several, a whole folder of people, zips) for the window
    start..end. Each file is read; the one with the most readings in the window is used, and gaps in it are filled
    from the next best. Files with no readings in the window (history, other people, other days) are ignored.
    Returns (5-min series or None, names of the files used)."""
    from clean import clean
    from common import to_5min
    from parse import read_frame
    found = []
    for f in unpack(files):
        if f.name.lower().endswith(tuple(PHOTO_TYPES)):
            continue
        try:
            df = read_frame(f)
        except Exception:
            continue
        if "glucose" not in df:
            continue
        g = df["glucose"].dropna()
        inside = g[(g.index >= start if start is not None else True) & (g.index <= end if end is not None else True)]
        if len(inside):
            found.append((len(inside), f.name, df[[c for c in COLUMNS if c in df]]))
    if not found:
        return None, []
    found.sort(key=lambda x: -x[0])
    best = found[0][0]
    used = [x for x in found if x[0] >= 0.2 * best]          # the best file, plus others that fill its gaps
    df = pd.concat([x[2] for x in used])
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return clean(to_5min(df)), [x[1] for x in used]


def sort_files(files):
    """Work out what each dropped file is: profile, history, meal(s), exercise, insulin doses, truth (validation) or
    photo. Columns are recognised in many spellings (parse.py), so CGMacros files and meter exports work too."""
    from parse import key, read_table, standard
    out = {"photo": None, "profile": None, "history": None, "meal": None, "exercise": None, "doses": None, "truth": None,
           "unknown": []}
    for f in files:
        name = f.name.lower()
        if name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            out["photo"] = f
            continue
        try:
            head = read_table(f, nrows=50)
        except Exception:
            out["unknown"].append(f.name)
            continue
        raw = {key(c) for c in head.columns}
        std = set(standard(head)[0].columns)
        if any(w in name for w in ("truth", "valid", "really", "check")):
            out["truth"] = f
        elif {"diabetestype", "a1c", "type"} & raw or "about" in name or "profile" in name:
            out["profile"] = f
        elif {"start", "minutes"} <= raw:
            out["exercise"] = f
        elif "units" in raw or ("bolus" in std and "glucose" not in std and "carbs" not in std):
            out["doses"] = f
        elif "glucose" in std:                    # several glucose files (a folder): the longest is the history
            if out["history"] is None or len(f.getvalue()) > len(out["history"].getvalue()):
                out["history"] = f
        elif "carbs" in std or "calories" in std:
            out["meal"] = f
        else:
            out["unknown"].append(f.name)
    return out


# ---------- Sample sets you can drag onto an uploader ----------

SAMPLES_JS = r"""
export default function(component) {
  const { data, parentElement } = component;
  const host = parentElement.querySelector('.sets');
  const full = u => new URL(u, document.baseURI).href;      // relative to the app: works at / and under a subpath (GitHub Pages)
  // Put files into a Streamlit file uploader: build File objects and hand them to its hidden input.
  window.__glkAdd = window.__glkAdd || (async (uploader, files) => {
    const dt = new DataTransfer();
    for (const f of files) {
      const r = await fetch(full(f.url));
      if (!r.ok) { console.warn('sample file missing', f.url); continue; }
      dt.items.add(new File([await r.blob()], f.name, { type: f.type }));
    }
    const input = uploader.querySelector('input[type=file]');
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
  if (!window.__glkDrop) {                      // one listener for the whole page: our cards dropped on any uploader
    window.__glkDrop = true;
    const zone = e => e.target.closest && e.target.closest('[data-testid="stFileUploader"]');
    document.addEventListener('dragover', e => {
      if (window.__glkFiles && zone(e)) { e.preventDefault(); e.stopPropagation(); e.dataTransfer.dropEffect = 'copy'; }
    }, true);
    document.addEventListener('drop', e => {
      const z = window.__glkFiles && zone(e);
      if (!z) return;
      e.preventDefault(); e.stopPropagation();
      const files = window.__glkFiles; window.__glkFiles = null;
      window.__glkAdd(z, files);
    }, true);
  }
  const icon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"
    stroke-linejoin="round"><path d="M12 4v11"/><path d="m7 10 5 5 5-5"/><path d="M5 20h14"/></svg>`;
  host.innerHTML = data.sets.map((s, i) => `
    <div class="card" draggable="true" data-i="${i}" title="Drag me onto the box (or click)">
      <div class="grip">⠿</div>
      <div class="txt"><div class="name">${s.title} <span style="color:var(--${s.color})">${s.sub}</span></div>
        <div class="files">${s.files.map(f => f.label).join(' · ')}</div></div>
      <a class="dl" href="${full(s.zip)}" download="${s.zip_name}" title="Download these files">${icon}</a>
    </div>`).join('');
  host.querySelectorAll('.card').forEach(c => {
    const s = data.sets[Number(c.dataset.i)];
    c.ondragstart = e => { window.__glkFiles = s.files; e.dataTransfer.setData('text/plain', s.title); e.dataTransfer.effectAllowed = 'copy'; c.classList.add('lift'); };
    c.ondragend = () => { c.classList.remove('lift'); setTimeout(() => { window.__glkFiles = null; }, 500); };
    c.onclick = e => {
      if (e.target.closest('.dl')) return;
      const target = document.querySelector('.st-key-' + data.target + ' [data-testid="stFileUploader"]') ||
                     document.querySelector('.st-key-' + data.target);
      if (target) { c.classList.add('sent'); window.__glkAdd(target, s.files); }
    };
  });
}
"""
SAMPLES_CSS = """
.sets { display: flex; flex-wrap: wrap; gap: .8rem; font-family: Libron, Georgia, serif; }
.card { display: flex; align-items: center; gap: .8rem; padding: .7rem .9rem; border: 1px dashed var(--tx3);
        border-radius: 10px; cursor: grab; flex: 1 1 260px; background: var(--bg2); transition: transform .12s, border-color .12s; }
.card:hover { border-color: var(--blue); transform: translateY(-1px); }
.card.lift { opacity: .55; }
.card.sent { border-style: solid; border-color: var(--green); }
.grip { color: var(--tx3); font-size: 1.3rem; line-height: 1; }
.txt { flex: 1; }
.name { font-size: 1.1rem; color: var(--tx); }
.name span { font-size: .85rem; margin-left: .3rem; }
.files { font-size: .8rem; color: var(--tx2); margin-top: .15rem; }
.dl { color: var(--tx2); display: inline-flex; padding: .35rem; border-radius: 8px; }
.dl:hover { color: var(--blue); background: var(--ui); }
.dl svg { width: 20px; height: 20px; }
"""
FILE_LABELS = {"photo": "photo", "about": "about you", "history": "glucose history", "meals": "meals", "meal": "meal",
               "insulin": "insulin plan", "exercise": "exercise", "really": "what really happened"}


def sample_sets(part, target, key):
    """Cards for the try_me sets (part = meal, meal_check, day or day_check): drag one onto any uploader, click it
    to add it to the uploader keyed `target`, or download it with the icon."""
    from urllib.parse import quote
    static = ROOT / "static" / "try_me"
    if not static.exists():
        return
    sets = []
    for person in sorted(p for p in static.iterdir() if p.is_dir()):
        folder = person / part
        if not folder.exists():
            continue
        name, kind = person.name.split(" (")
        group = next((g for g, n in TYPE_NAME.items() if n == kind.rstrip(")")), "t1d")
        url = lambda p: "app/static/" + quote(p.relative_to(ROOT / "static").as_posix())
        files = [{"name": f.name, "url": url(f), "type": "image/jpeg" if f.suffix == ".jpg" else "text/csv",
                  "label": next((l for w, l in FILE_LABELS.items() if w in f.name), f.stem)}
                 for f in sorted(folder.iterdir())]
        zip_ = person / f"{person.name} {part}.zip"
        sets.append({"title": name, "sub": TYPE_NAME[group], "color": TYPE_COLOR[group], "files": files,
                     "zip": url(zip_), "zip_name": zip_.name})
    if "_samples" not in st.session_state:
        st.session_state["_samples"] = st.components.v2.component("sample_sets", html="<div class='sets'></div>",
                                                                  css=SAMPLES_CSS, js=SAMPLES_JS)
    st.session_state["_samples"](key=key, data={"sets": sets, "target": target})


# ---------- Exercise and insulin in a plan ----------

def at(day, clock):
    """A day and a clock time from an edited table -> Timestamp. The browser version hands times back as text."""
    if isinstance(clock, str):
        clock = pd.Timestamp(f"2000-01-01 {clock}").time()
    elif hasattr(clock, "time") and not isinstance(clock, type(pd.Timestamp(0).time())):
        clock = clock.time()
    return pd.Timestamp.combine(day, clock)


def exercise_editor(key, day, rows=None):
    """Exercise sessions: start (clock time), minutes, how hard. -> [(time, minutes, intensity)] on `day`."""
    df = pd.DataFrame(rows if rows is not None else [], columns=["start", "minutes", "intensity"])
    df["start"] = pd.Series([pd.Timestamp(t).time() for t in df["start"]], dtype=object)
    out = st.data_editor(df, num_rows="dynamic", key=key, width="stretch", column_config={
        "start": st.column_config.TimeColumn("starts at", format="HH:mm", step=300),
        "minutes": st.column_config.NumberColumn("minutes", min_value=5, max_value=300, step=5, default=30),
        "intensity": st.column_config.SelectboxColumn("how hard", options=INTENSITY, default="moderate")})
    return [(at(day, r["start"]), int(r["minutes"]), r["intensity"]) for _, r in out.iterrows()
            if pd.notna(r["start"]) and pd.notna(r["minutes"]) and r["intensity"] in INTENSITY]


def insulin_editor(key, day, rows=None):
    """Insulin doses not taken with a meal: clock time, units. -> [(time, units)] on `day`."""
    df = pd.DataFrame(rows if rows is not None else [], columns=["time", "units"])
    df["time"] = pd.Series([pd.Timestamp(t).time() for t in df["time"]], dtype=object)
    out = st.data_editor(df, num_rows="dynamic", key=key, width="stretch", column_config={
        "time": st.column_config.TimeColumn("time", format="HH:mm", step=300),
        "units": st.column_config.NumberColumn("units", min_value=0.0, max_value=50.0, step=0.5)})
    return [(at(day, r["time"]), float(r["units"])) for _, r in out.iterrows()
            if pd.notna(r["time"]) and pd.notna(r["units"])]


def read_meals(file):
    """A meal file in any common shape -> list of {time, meal, carbs, protein, fat, fiber, calories, insulin}."""
    from parse import key, read_frame, read_table
    if file is None:
        return []
    df = read_frame(file)
    raw = read_table(file)
    name_col = next((c for c in raw.columns if key(c) in ("meal", "mealtype", "name", "food", "description")), None)
    names = raw[name_col].astype(str).tolist() if name_col is not None else [""] * len(raw)
    out = []
    for (t, r), name in zip(df.iterrows(), names):
        m = {"time": pd.Timestamp(t), "meal": name if name not in ("", "nan") else meal_name(pd.Timestamp(t)),
             **{c: float(np.nan_to_num(r.get(c, 0.0))) for c in NUTRIENTS}, "insulin": float(np.nan_to_num(r.get("bolus", 0.0)))}
        if any(m[c] > 0 for c in NUTRIENTS):
            out.append(m)
    return out


def meal_name(t):
    h = t.hour + t.minute / 60
    return "Snack" if h < 4 or h >= 21.5 else "Breakfast" if h < 11 else "Lunch" if h < 16 else "Dinner"


def meals_in(history, start, end):
    """Meals recorded inside a history file between start and end (for testing on a day the file already has)."""
    rows = history[(history.index >= start) & (history.index < end)]
    food = rows[(rows[NUTRIENTS].fillna(0) > 0).any(axis=1)]
    return [{"time": t, "meal": meal_name(t), **{c: float(r[c]) for c in NUTRIENTS}, "insulin": float(r.get("bolus", 0) or 0)}
            for t, r in food.iterrows()]


def csv_rows(file):
    """An uploaded or local CSV as a list of row dicts ([] if none)."""
    if file is None:
        return []
    from parse import read_table
    df = read_table(file)
    df.columns = df.columns.str.strip().str.lower()
    return df.to_dict("records")


# ---------- Learning a person and predicting ----------

@st.cache_resource
def _shared_store(version):
    """People learned once for every visitor (the demo people)."""
    return {}


@st.cache_resource
def _classifier():
    return load_classifier()


def route(history, info):
    """Just which type and backend (a tiny classifier, no model fitting): enough for step 1."""
    return classify(info, history, _classifier())


def best_kind(bundle, backend):
    """Which of the two models a backend uses: the one with the lower error on the tuning days (days 1-5 -> 6-7 of
    the training people), never on test data."""
    scores = {k: getattr(m, "tune_rmse", None) for (b, k), m in bundle.items() if b == backend}
    if any(v is None for v in scores.values()):                    # older weights: read the saved tuning results
        path = ROOT / "results" / "tuning.csv"
        if path.exists():
            t = pd.read_csv(path)
            scores = t[t["backend"] == backend].groupby("kind")["rmse"].min().to_dict()
    scores = {k: v for k, v in scores.items() if v is not None}
    return min(scores, key=scores.get) if scores else "regressor"


def learn(history, info, key, kind=None, exclude=None, shared=False):
    """Route the person to a backend and fit their coefficients (cached for this session, or for everyone if shared).
    kind = None: the backend's better model on the tuning days.
    exclude = a dataset id: take that person out of the population and group priors (keeps a demo never-seen)."""
    store = _shared_store(code_version()) if shared else st.session_state.setdefault("_people", {})
    k = (key, kind, exclude, code_version())
    if k not in store:
        bundle, clf = load_models(code_version())
        route = classify(info, history, clf)
        chosen = kind if kind in ("regressor", "overlay") else best_kind(bundle, route["backend"])
        model = bundle[(route["backend"], chosen)]
        base = baseline(history)
        group = route["group"] if route["group"] in model.groups else next(iter(model.groups))
        prior = model.prior(group, exclude=exclude if exclude in getattr(model, "person_stats", {}) else None)
        store[k] = {"route": route, "backend": route["backend"], "group": route["group"], "base": base, "model": model,
                    "post": model.posterior(history, group, base, prior=prior), "prior": prior, "fit_group": group,
                    "kind": chosen}
    return store[k]


def contribute(history, later, info, page):
    """Save a validated prediction (history before it + what really happened) to data/contributed/, so the next
    retraining can learn from it. Kept on this machine; nothing is sent anywhere."""
    folder = ROOT / "data" / "contributed" / f"{pd.Timestamp.now():%Y%m%d-%H%M%S}-{page}"
    folder.mkdir(parents=True, exist_ok=True)
    cols = [c for c in COLUMNS if c in history]
    history[cols].rename(columns={"glucose": "glucose_clean"}).assign(glucose=history["glucose_raw"]).rename_axis("time").to_csv(folder / "history.csv")
    later.reindex(columns=[c for c in COLUMNS if c in later] + (["glucose_raw"] if "glucose_raw" in later else []))         .assign(glucose=later.get("glucose_raw", later.get("glucose"))).drop(columns=["glucose_raw"], errors="ignore")         .rename_axis("time").to_csv(folder / "what_really_happened.csv")
    pd.DataFrame([info]).to_csv(folder / "about_me.csv", index=False)


REPLAY_MEALS, BAND_STEPS, POP_SEED = 20, 36, 10       # meals replayed, band steps (3 h), population misses mixed in


def track_record(person, history):
    """Replay the person's own history meal by meal: for each of their last REPLAY_MEALS meals (from day 2 on), learn from
    everything before it, predict the 3 h after it and note the miss against the raw sensor. Cached on the person.
    Returns {'mae', 'rmse', 'meals', 'resid' (meals x 36 steps: truth - prediction)}, or None without usable meals.
    The misses give the honest "off by" number and the person's own band (personal_band)."""
    if "_track" in person:
        return person["_track"]
    model, base, group, prior = person["model"], person["base"], person["fit_group"], person.get("prior")
    g = history["glucose"].to_numpy(float)
    raw = history["glucose_raw"].to_numpy(float)
    first = np.searchsorted(history.index, history.index[0] + pd.Timedelta(days=1))
    meals = [a for a in np.flatnonzero(history["carbs"].to_numpy() > 0)
             if a >= first and a + 24 < len(g) and np.isfinite(g[a]) and np.isfinite(raw[a + 1: a + 25]).sum() >= 20]
    meals = meals[-REPLAY_MEALS:]
    out = None
    if meals:
        rows = model.rows(history, base)                  # built once; each replay step only hides the future
        meal_X = {int(a): x for a, x in zip(rows[0][0], rows[0][1])} if rows else {}    # their inputs, reused
        resid = np.full((len(meals), BAND_STEPS), np.nan)
        prior = model.prior(group) if prior is None else prior
        seen = model.stats_until(rows, meals)              # what each replayed meal may learn from, in one pass
        for k, a in enumerate(meals):
            coef = solve(seen[k], prior, model.person_strength, model.scale)
            p = (g[a] + meal_X[a] @ coef if a in meal_X else model.predict_from(history, a, coef, base)[0])[:BAND_STEPS]
            stop = min(a + 1 + BAND_STEPS, len(raw))
            resid[k, : stop - a - 1] = raw[a + 1: stop] - p[: stop - a - 1]
        e = resid[np.isfinite(resid)]
        out = {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "meals": len(meals), "resid": resid}
    person["_track"] = out
    return out


# Which personal band each backend uses: the tightest one that still held the truth >= 80 % of the time on later meals
# of never-seen people (fitted on their replayed days 2-7, checked on day 8+).
#   "pooled": per-step percentiles of the person's own replayed misses + 10 borrowed  (metabolic: 80.1 %, 48.0 wide)
#   "scaled": everyone's band shape, one personal scale fitted to 80 % of their own readings (metabolic: 83.7 %, 54.7)
#   None: the model's calibrated band (type 1, where the personal versions held too little)
BAND_METHOD = {"metabolic": "pooled", "t1d": None}


def personal_band(person, pred, level=0.8):
    """The person's own band for the first 3 h, by BAND_METHOD for their backend. Returns (lo, hi) or None."""
    method = BAND_METHOD.get(person["backend"])
    track, pop = person.get("_track"), getattr(person["model"], "pop_resid", None)
    if method is None or not track or pop is None or track["meals"] < 3:
        return None
    q = [(1 - level) / 2, (1 + level) / 2]
    own = track["resid"]
    if method == "pooled":
        rng = np.random.default_rng(0)
        off = np.nanquantile(np.concatenate([own, pop[rng.choice(len(pop), POP_SEED, replace=False), :BAND_STEPS]]), q, axis=0)
    else:
        shape = np.nanquantile(pop[:, :BAND_STEPS], q, axis=0)
        ks = np.linspace(0.2, 4.0, 381)
        inside = np.array([np.nanmean((own >= k * shape[0]) & (own <= k * shape[1])) for k in ks])
        k = float(ks[np.argmax(inside >= level)]) if inside.max() >= level else float(ks[-1])
        off = k * shape
    n = min(len(pred), BAND_STEPS)
    p = pred["pred"].to_numpy()
    lo, hi = pred["lo"].to_numpy().copy(), pred["hi"].to_numpy().copy()
    lo[:n], hi[:n] = p[:n] + off[0][:n], p[:n] + off[1][:n]
    if len(pred) > n:                                     # after 3 h: the model's band, never narrower than at 3 h
        lo[n:] = np.minimum(lo[n:], p[n:] + off[0][n - 1])
        hi[n:] = np.maximum(hi[n:], p[n:] + off[1][n - 1])
    return lo, hi


def predict_plan(person, history, start, end, meals=(), sessions=(), doses=()):
    """Glucose from `start` to `end` given the plan (meals, exercise sessions, insulin doses), with its calibrated sd."""
    s = plan_series(history, start, end, person["backend"], meals, sessions, doses)
    start = pd.Timestamp(start).floor("5min")
    p, sd = predict_window(person["model"], s[COLUMNS], person["post"], start, pd.Timestamp(end), person["base"])
    model = person["model"]
    steps = np.arange(1, len(p) + 1)
    lo, hi = model.band(p.to_numpy(), sd.to_numpy(), steps) if hasattr(model, "band") else (p - Z80 * sd, p + Z80 * sd)
    return pd.DataFrame({"pred": p, "sd": sd, "lo": lo, "hi": hi}, index=p.index)


def predict_after_meal(person, history, meal, hours=AFTER):
    """Glucose for `hours` after the meal. meal may carry 'doses' [(time, units)] and 'sessions' [(time, minutes,
    intensity)] for the hours after it. DataFrame(pred, sd)."""
    t0 = pd.Timestamp(meal["time"]).floor("5min")
    pred = predict_plan(person, history, t0, t0 + hours, [meal], meal.get("sessions", []), meal.get("doses", []))
    track_record(person, history)                         # the person's own misses set their band
    band = personal_band(person, pred)
    if band is not None:
        pred["lo"], pred["hi"] = band
        pred.attrs["band"] = "personal"
    return pred


def score(pred, truth):
    """How close: RMSE, mean absolute error (mg/dL and %), and how often the truth sits inside the 80 % band."""
    both = pd.concat([pred["pred"], pred["lo"], pred["hi"], truth], axis=1, keys=["p", "lo", "hi", "a"]).dropna()
    if both.empty:
        return None
    err = both["p"] - both["a"]
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "mape": float(np.mean(np.abs(err) / both["a"]) * 100),
            "inside": float(((both["a"] >= both["lo"]) & (both["a"] <= both["hi"])).mean()), "n": len(both)}


# ---------- Plots ----------

def show(fig, height, **kw):
    titled = "title" in kw
    layout = dict(height=height + (30 if titled else 10), font=dict(family=FONT, size=13), hovermode="x unified",
                  margin=dict(l=50, r=16, t=78 if titled else 44, b=10),
                  legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))   # above: never on the axis
    if titled and isinstance(kw["title"], str):
        kw["title"] = dict(text=kw["title"], y=1, yanchor="top", yref="container", x=0, xanchor="left", pad=dict(t=8))
    layout.update(kw)
    fig.update_layout(**layout)
    st.plotly_chart(fig, width="stretch", theme="streamlit", config={"displayModeBar": False})


def glucose_axis(*values, room=0):
    """Always 60 to 200 mg/dL, widened only as far as the data or band reach (to the next 5), labels every 20.
    room: extra space below for event markers. Returns (yaxis dict, bottom)."""
    vals = np.concatenate([np.asarray(v, float).ravel() for v in values if v is not None and len(v)]) if values else np.array([])
    vals = vals[np.isfinite(vals)]
    lo = min(60.0, 5 * np.floor(vals.min() / 5)) if len(vals) else 60.0
    hi = max(200.0, 5 * np.ceil(vals.max() / 5)) if len(vals) else 200.0
    lo -= room
    return dict(range=[lo, hi], dtick=20, title="glucose (mg/dL)"), lo


def unhealthy_bands(fig):
    """Grey bands where glucose is out of the healthy range: below 70 and above 180 mg/dL."""
    P = palette()
    for lo, hi_ in ((0, 70), (180, 600)):
        fig.add_hrect(y0=lo, y1=hi_, fillcolor=rgba(P["tx3"], 0.18), line_width=0, layer="below")


def rgba(color, alpha):
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def meal_plot(history, meal, pred=None, truth=None, hours_before=5, height=440):
    """Known glucose before the meal, the prediction with its band, and (optionally) what really happened."""
    P = palette()
    t0 = pd.Timestamp(meal["time"])
    fig = go.Figure()
    unhealthy_bands(fig)
    known = history["glucose_raw"][(history.index >= t0 - pd.Timedelta(hours=hours_before)) & (history.index < t0)]
    fig.add_scatter(x=known.index, y=known, name="your glucose so far", line=dict(color=P["red"], width=2.4),
                    hovertemplate="%{y:.0f} mg/dL")
    if pred is not None:
        fig.add_scatter(x=pred.index, y=pred["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=pred.index, y=pred["lo"], fill="tonexty", line=dict(width=0),
                        fillcolor=rgba(P["magenta"], 0.16), name="where it will most likely be (80 %)", hoverinfo="skip")
        fig.add_scatter(x=pred.index, y=pred["pred"], name="our prediction", line=dict(color=P["magenta"], width=3, dash="dash"),
                        hovertemplate="%{y:.0f} mg/dL")
    if truth is not None:
        fig.add_scatter(x=truth.index, y=truth, name="what really happened", line=dict(color=P["green"], width=3),
                        hovertemplate="%{y:.0f} mg/dL")
    fig.add_vline(x=t0, line=dict(color=P["yellow"], dash="dot", width=1.5))
    fig.add_annotation(x=t0, y=1, yref="paper", yanchor="bottom", showarrow=False, font=dict(color=P["yellow"], size=13),
                       text=f"{meal.get('name', 'meal')} · {meal.get('carbs', 0):.0f} g carbs")
    for mins in (30, 60, 90, 120):
        fig.add_vline(x=t0 + pd.Timedelta(minutes=mins), line=dict(color=P["tx3"], width=1))
    yaxis, _ = glucose_axis(known, *([pred["lo"], pred["hi"]] if pred is not None else []), truth)
    show(fig, height, yaxis=yaxis, xaxis_range=[t0 - pd.Timedelta(hours=hours_before), t0 + AFTER + pd.Timedelta(minutes=10)])


def history_plot(history, until=None, days=2, height=300):
    """The last couple of days of glucose with meals marked."""
    P = palette()
    until = until or history.index[-1]
    h = history[(history.index >= until - pd.Timedelta(days=days)) & (history.index <= until)]
    fig = go.Figure()
    unhealthy_bands(fig)
    fig.add_scatter(x=h.index, y=h["glucose_raw"], name="glucose", line=dict(color=P["red"], width=1.8), hovertemplate="%{y:.0f} mg/dL")
    meals = h[h["carbs"] > 0]
    yaxis, bottom = glucose_axis(h["glucose_raw"], room=15)
    fig.add_scatter(x=meals.index, y=[bottom + 10] * len(meals), mode="markers", name="meals",
                    marker=dict(color=P["yellow"], size=10, line=dict(width=1, color=P["bg"])),
                    text=[f"{c:.0f} g carbs" for c in meals["carbs"]], hovertemplate="%{text}")
    if h["bolus"].sum() > 0:
        b = h[h["bolus"] > 0]
        fig.add_scatter(x=b.index, y=[bottom + 3] * len(b), mode="markers", name="insulin",
                        marker=dict(color=P["orange"], symbol="triangle-up", size=9),
                        text=[f"{u:.1f} U" for u in b["bolus"]], hovertemplate="%{text}")
    show(fig, height, yaxis={**yaxis, "title": "mg/dL"}, xaxis=dict(nticks=6, tickformat="%a %H:%M", tickangle=0))


def day_plot(history, pred, meals=(), sessions=(), doses=(), truth=None, height=480):
    """The day ahead: the last hours of known glucose, the prediction with its band, meals, exercise and insulin."""
    P = palette()
    start = pred.index[0]
    fig = go.Figure()
    known = history["glucose_raw"][(history.index >= start - pd.Timedelta(hours=6)) & (history.index < start)]
    fig.add_scatter(x=known.index, y=known, name="your glucose so far", line=dict(color=P["red"], width=2.2),
                    hovertemplate="%{y:.0f} mg/dL")
    unhealthy_bands(fig)
    fig.add_scatter(x=pred.index, y=pred["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=pred.index, y=pred["lo"], fill="tonexty", line=dict(width=0),
                    fillcolor=rgba(P["magenta"], 0.16), name="where it will most likely be (80 %)", hoverinfo="skip")
    fig.add_scatter(x=pred.index, y=pred["pred"], name="our prediction", line=dict(color=P["magenta"], width=2.6, dash="dash"),
                    hovertemplate="%{y:.0f} mg/dL")
    if truth is not None:
        tr = truth[(truth.index >= start) & (truth.index <= pred.index[-1])]
        fig.add_scatter(x=tr.index, y=tr, name="what really happened", line=dict(color=P["green"], width=2.6),
                        hovertemplate="%{y:.0f} mg/dL")
    for t, mins, inten in sessions:
        fig.add_vrect(x0=t, x1=t + pd.Timedelta(minutes=mins), fillcolor=rgba(P["purple"], 0.18), line_width=0)
    yaxis, bottom = glucose_axis(known, pred["lo"], pred["hi"], truth if truth is not None else None, room=25)
    y_marks = bottom + 18
    if meals:
        fig.add_scatter(x=[m["time"] for m in meals], y=[y_marks] * len(meals), mode="markers", name="meals",
                        marker=dict(color=P["yellow"], size=12, line=dict(width=1, color=P["bg"])),
                        text=[f"{m.get('meal', 'meal')}: {m.get('carbs', 0):.0f} g carbs" for m in meals], hovertemplate="%{text}")
    shots = [(m["time"], m.get("insulin", 0)) for m in meals if (m.get("insulin", 0) or 0) > 0] + list(doses)
    if shots:
        fig.add_scatter(x=[t for t, _ in shots], y=[y_marks - 12] * len(shots), mode="markers", name="insulin",
                        marker=dict(color=P["orange"], symbol="triangle-up", size=10),
                        text=[f"{u:.1f} U" for _, u in shots], hovertemplate="%{text}")
    if sessions:
        fig.add_scatter(x=[t for t, _, _ in sessions], y=[y_marks] * len(sessions), mode="markers", name="exercise",
                        marker=dict(color=P["purple"], symbol="diamond", size=10),
                        text=[f"{m} min, {i}" for _, m, i in sessions], hovertemplate="%{text}")
    show(fig, height, yaxis=yaxis, xaxis_range=[start - pd.Timedelta(hours=6), pred.index[-1]])


def day_insights(pred, meals=(), sessions=(), doses=(), backend="metabolic"):
    """A few plain sentences about the day: (colour, text)."""
    p = pred["pred"]
    out = []
    start = p.index[0]
    if p.idxmax() - start < pd.Timedelta(hours=1) and len(p) > 36:      # starting high is not the day's peak
        settle = p.iloc[:36].idxmin()
        out.append(("orange", f"Starts high, around <b>{p.iloc[0]:.0f} mg/dL</b>, and settles to about {p[settle]:.0f} by {settle:%H:%M}."))
        p_rest = p[p.index >= start + pd.Timedelta(hours=2)]
    else:
        p_rest = p
    t_peak = p_rest.idxmax()
    before = [m for m in meals if m["time"] <= t_peak and t_peak - m["time"] <= pd.Timedelta(hours=3)]
    cause = f", about {(t_peak - before[-1]['time']).total_seconds() / 60:.0f} min after {before[-1].get('meal', 'a meal').lower()}" if before else ""
    out.append(("magenta", f"Highest around <b>{t_peak:%H:%M}</b>: about <b>{p_rest.max():.0f} mg/dL</b>{cause}."))
    rises = []
    for m in meals:
        w = p[(p.index >= m["time"]) & (p.index <= m["time"] + AFTER)]
        if len(w) > 1:
            rises.append((w.max() - w.iloc[0], m))
    if rises:
        r, m = max(rises, key=lambda x: x[0])
        out.append(("yellow", f"Biggest spike: <b>{m.get('meal', 'meal')} at {m['time']:%H:%M}</b> "
                              f"({m.get('carbs', 0):.0f} g carbs), up about <b>{r:.0f} mg/dL</b>."))
    in_range = float(((p >= 70) & (p <= 180)).mean())
    out.append(("green" if in_range >= 0.7 else "orange", f"About <b>{in_range:.0%}</b> of the day in the healthy range (70 to 180 mg/dL)."))
    t_low = p.idxmin()
    if p.min() < 80:                    # the prediction itself heads low (the band alone widens through any day)
        why = []
        shots = [(m["time"], m.get("insulin", 0) or 0) for m in meals] + list(doses)
        units = sum(u for t, u in shots if t_low - pd.Timedelta(hours=4) <= t <= t_low and u > 0)
        if backend == "t1d" and units >= 1:
            why.append(f"{units:.1f} units of insulin in the 4 hours before")
        ex = [x for x in sessions if x[0] - pd.Timedelta(hours=1) <= t_low <= x[0] + pd.Timedelta(minutes=x[1] + 120)]
        if ex:
            why.append(f"{ex[0][1]} minutes of {ex[0][2]} exercise")
        what = f"A low is <b>likely</b> around <b>{t_low:%H:%M}</b>, down to about {p.min():.0f} mg/dL" if p.min() < 70 else                f"Dips close to low around <b>{t_low:%H:%M}</b>, about {p.min():.0f} mg/dL"
        out.append(("red", what + (f", after {' and '.join(why)}" if why else "") + ". Worth keeping a snack close."))
    else:
        out.append(("blue", f"Lowest around {t_low:%H:%M}, about {p.min():.0f} mg/dL. No lows expected."))
    for t, mins, inten in sessions:
        w = p[(p.index >= t) & (p.index <= t + pd.Timedelta(minutes=mins + 60))]
        if len(w) > 2:
            out.append(("purple", f"Your {mins}-minute {inten} exercise at {t:%H:%M}: glucose "
                                  f"{'drops' if w.iloc[-1] < w.iloc[0] else 'rises'} about {abs(w.iloc[-1] - w.iloc[0]):.0f} mg/dL by an hour after it ends."))
    high = float((p > 180).sum() * 5 / 60)
    if high >= 0.5:
        out.append(("orange", f"Above 180 mg/dL for <b>{'about an hour' if high < 1.5 else f'about {high:.1f} hours'}</b>, first around {p[p > 180].index[0]:%H:%M}."))
    return out


def rise_text(pred, g0, usual=None):
    """The standard headline: 'Increasing about 40 mg/dL (from 70), peaking about 90 min after eating, at 110 (88 to 136).'
    Falling or flat curves get the matching wording."""
    p = pred["pred"]
    rng = lambda t: f" ({pred['lo'][t]:.0f} to {pred['hi'][t]:.0f})" if "lo" in pred else ""
    rise, fall = p.max() - g0, g0 - p.min()
    if rise < 5 and fall < 5:
        return f"Staying about level, between <b class='c-magenta'>{p.min():.0f}</b> and <b class='c-magenta'>{p.max():.0f}</b> mg/dL."
    vs = lambda v: f" ({v - usual:+.0f} vs your usual {usual:.0f})" if usual is not None else ""
    if fall > rise:
        t, i = p.idxmin(), int(p.values.argmin())
        when = "still falling at 3 h" if i >= len(p) - 3 else f"lowest about {i * 5} min after eating"
        return (f"Decreasing about {fall:.0f} mg/dL{vs(p.min())}, {when}, at "
                f"<b class='c-magenta'>{p.min():.0f}</b>{rng(t)}.")
    t, i = p.idxmax(), int(p.values.argmax())
    when = "still rising at 3 h" if i >= len(p) - 3 else f"peaking about {i * 5} min after eating"
    return (f"Increasing about {rise:.0f} mg/dL{vs(p.max())}, {when}, at "
            f"<b class='c-magenta'>{p.max():.0f}</b>{rng(t)}.")


def meal_commentary(pred, history, meal, most=2):
    """At most `most` notes worth reading, most important first: [(colour, text)]. Every number comes from the
    prediction, its band, or the person's own past meals; nothing is said unless it stands out."""
    p, lo, hi = pred["pred"], pred["lo"], pred["hi"]
    before = history["glucose_raw"].dropna()
    g0 = float(before.iloc[-1]) if len(before) else float(p.iloc[0])
    t0 = pd.Timestamp(meal["time"])
    notes = []                                         # (priority, colour, text): lower = more important
    later = p[p.index >= t0 + pd.Timedelta(minutes=20)]     # where it starts isn't a low it is heading for
    lo_later = lo[lo.index >= t0 + pd.Timedelta(minutes=20)]
    if len(later) and later.min() < 70:
        notes.append((0, "red", f"<b>Likely low</b> around {later.idxmin():%H:%M} (about {later.min():.0f}). Keep a snack close."))
    elif len(later) and later.min() < min(85, g0 - 5) and lo_later.min() < 70:   # heading down toward low, not just the band
        notes.append((1, "red", f"<b>Possible low</b> around {later.idxmin():%H:%M} (about {later.min():.0f})."))
    above = int((p > 180).sum() * 5)
    if above >= 15:
        notes.append((2, "orange", f"<b>Above 180 for about {above} min.</b>"))
    rise = float(p.max() - g0)
    carbs = float(meal.get("carbs", 0) or 0)
    if carbs >= 10:                                    # unusual for this person? (their past meals with similar carbs)
        g, c = history["glucose_raw"].to_numpy(float), history["carbs"].to_numpy(float)
        past = [np.nanmax(g[a: a + 25]) - g[a] for a in np.flatnonzero((c >= carbs * 0.6) & (c <= carbs * 1.4))
                if a + 25 <= len(g) and np.isfinite(g[a]) and np.isfinite(g[a: a + 25]).sum() >= 20]
        if len(past) >= 3:
            typical = float(np.median(past))
            if abs(rise - typical) > max(15, 0.3 * typical):
                notes.append((3, "yellow", f"{'Bigger' if rise > typical else 'Smaller'} than usual ({rise - typical:+.0f}): "
                                           f"similar meals usually rise about {typical:.0f} mg/dL."))
    after = p[p.index > p.idxmax()]
    if rise >= 20 and len(after) and after.min() > g0 + 10:
        notes.append((4, "orange", f"Still about {p.iloc[-1] - g0:.0f} above your start after 3 h."))
    elif rise >= 20 and len(after):
        back = after[after <= g0 + 10]
        notes.append((5, "green", f"Back to where you started by {back.index[0]:%H:%M}."))
    return [(c, t) for _, c, t in sorted(notes)[:most]]


def commentary_html(items):
    return "".join(f"<div style='font-size:1.08rem;line-height:1.55;margin:.25rem 0'><span class='c-{c}'>&#9679;</span> {t}</div>"
                   for c, t in items)


# ---------- Clickable photos ----------

PICK_JS = r"""
export default function(component) {
  const { data, setTriggerValue, parentElement } = component;
  const host = parentElement.querySelector('.picks');
  host.innerHTML = data.items.map((it, i) => `
    <button class="pick${it.chosen ? ' chosen' : ''}" data-i="${i}" title="Use this meal's logged numbers">
      ${it.src ? `<img src="${it.src}">` : ''}${it.title ? `<b>${it.title}</b>` : ''}<span>${it.caption}</span></button>`).join('');
  host.querySelectorAll('.pick').forEach(b => { b.onclick = () => setTriggerValue('pick', Number(b.dataset.i)); });
}
"""
PICK_CSS = """
.picks { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: .6rem; font-family: Libron, Georgia, serif; }
.pick { all: unset; cursor: pointer; border: 2px solid transparent; border-radius: 10px; overflow: hidden;
        background: var(--bg2); display: flex; flex-direction: column; transition: border-color .12s, transform .12s; }
.pick:hover { border-color: var(--blue); transform: translateY(-1px); }
.pick.chosen { border-color: var(--green); }
.pick img { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; display: block; }
.pick span { font-size: .8rem; color: var(--tx2); padding: .3rem .45rem .4rem; line-height: 1.3; }
.pick b { font-size: .95rem; font-weight: 400; color: var(--tx); padding: .5rem .5rem 0; line-height: 1.3; }
"""


@st.cache_data(show_spinner=False, max_entries=256)
def thumb(path, size=240):
    """A small JPEG of a photo as a data URI (for clickable pictures)."""
    import base64
    import io as io_
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    buf = io_.BytesIO()
    im.save(buf, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _picker(items, key):
    if "_picker" not in st.session_state:
        st.session_state["_picker"] = st.components.v2.component("photo_picker", html="<div class='picks'></div>",
                                                                 css=PICK_CSS, js=PICK_JS)
    res = st.session_state["_picker"](key=key, data={"items": items}, on_pick_change=lambda: None)
    return getattr(res, "pick", None)


def image_picker(items, key):
    """Clickable photos: items = [(path, caption, chosen)]. Returns the index clicked this run, or None."""
    return _picker([{"src": thumb(p), "caption": c, "chosen": ch} for p, c, ch in items], key)


def option_picker(items, key):
    """Clickable cards without pictures: items = [(title, caption, chosen)]. Returns the index clicked, or None."""
    return _picker([{"src": None, "title": t, "caption": c, "chosen": ch} for t, c, ch in items], key)


def nutrition_line(n):
    """'34 g carbs · 5 g protein · 5 g fat · 2 g fiber · 200 kcal'"""
    return " · ".join(f"{n.get(f, 0):.0f}{u}" for f, u in (("carbs", " g carbs"), ("protein", " g protein"), ("fat", " g fat"),
                                                          ("fiber", " g fiber"), ("calories", " kcal")))


def accuracy_note(pred, track, who="you"):
    """How good the prediction usually is, in two numbers: how far the real readings typically sit from the predicted
    line (RMSE, from replaying the person's own recent meals) and how many fall inside the band."""
    you = who == "you"
    pron = "your" if you else f"{who}'s"
    rmse = (f"Over {pron} last {track['meals']} meals, the real readings were typically <b>{track['rmse']:.0f} mg/dL</b> "
            "above or below our predicted line (RMSE). " if track else "")
    band = (f"The shaded band is {pron} own and holds about 8 in 10 readings." if pred.attrs.get("band") == "personal"
            else "The shaded band holds about 8 in 10 readings.")
    return f"<p class='small'>{rmse}{band}</p>"


def day_headline(pred):
    """One sentence for a whole day: highest, lowest, time in range."""
    p = pred["pred"]
    th, tl = p.idxmax(), p.idxmin()
    rng = lambda t: f" ({pred['lo'][t]:.0f} to {pred['hi'][t]:.0f})"
    in_range = float(((p >= 70) & (p <= 180)).mean())
    return (f"Highest about <b class='c-magenta'>{p.max():.0f}</b>{rng(th)} around {th:%H:%M}, lowest about "
            f"<b class='c-magenta'>{p.min():.0f}</b>{rng(tl)} around {tl:%H:%M}; {in_range:.0%} of the day between 70 and 180.")
