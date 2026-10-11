"""Food photo -> nutrition (carbs, protein, fat, fiber, calories). A separate model; its output is just the meal
input for either backend. Everything here is open: CLIP (MIT licence) and USDA FNDDS (public domain).

1. Recognise: CLIP ViT-L/14 (openai/clip-vit-large-patch14; run with OpenVINO on the Intel GPU when there is one)
   compares the photo with a text description of every
   food in USDA FNDDS 2021-2023 ("a photo of burrito, a type of food"), about 5,000 foods, with no extra training
   (zero-shot). The top matches and how sure it is are shown to the user.
2. Look up: FNDDS gives nutrients per 100 g for each food and the typical amount people eat when they don't say how
   much ("quantity not specified", from the US national diet survey). Nutrition = the probability-weighted
   average of the top matches at their typical portion.
3. Photo memory: the photo is also compared with meal photos logged before with known nutrition (the first week of
   all 45 CGMacros people). No names or labels: CLIP turns each photo into 768 numbers describing how it looks, and
   "close" means those numbers point the same way (cosine similarity, 1 = identical). The 5 most similar meals'
   logged nutrition is averaged (closer counts more), and the closer the best match, the more the memory counts
   against the lookup. People eat the same kinds of meals again and again, so a near-identical photo is good evidence.
How much to trust memory vs lookup, and a per-nutrient scale for the lookup, are set on CGMacros days 1-5 -> 6-7
photos; the test below is on day 8+ photos only. The portion slider on the site then scales the result.

    python models/photo2nutrition.py      builds the FNDDS table, embeds photos, tunes, tests -> results/
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from common import DATA, ROOT

MODEL = "openai/clip-vit-large-patch14"
FNDDS = DATA / "fndds" / "FoodData_Central_survey_food_csv_2024-10-31"
TABLE = DATA / "fndds_foods.csv"
CACHE = DATA / "cache"
TEXT = CACHE / "fndds_text_L14.npy"
MEMORY = CACHE / "photo_memory_L14.npz"
PHOTOS = CACHE / "cgmacros_photos_L14.npz"
SETTINGS = CACHE / "photo_settings.json"
NUTRIENTS = ["carbs", "protein", "fat", "fiber", "calories"]
NUTRIENT_IDS = {205: "carbs", 203: "protein", 204: "fat", 291: "fiber", 208: "calories"}     # FNDDS nutrient numbers
SKIP = ("baby", "infant", "formula", "human milk", "nutritional", "supplement")      # not things people photograph
DEFAULTS = {"temperature": 100.0, "top": 5, "memory_center": 0.85, "memory_width": 0.03,
            "scale": {n: 1.0 for n in NUTRIENTS}}


# ---------- USDA FNDDS table ----------

def build_table():
    """One row per FNDDS food: name, category, nutrients for its typical portion, the portion in words and grams."""
    food = pd.read_csv(FNDDS / "food.csv")
    cats = pd.read_csv(FNDDS / "wweia_food_category.csv").set_index("wweia_food_category")["wweia_food_category_description"]
    food["category"] = food["food_category_id"].map(cats).fillna("")
    keep = ~(food["description"].str.lower().str.contains("|".join(SKIP)) | food["category"].str.lower().str.contains("|".join(SKIP)))
    food = food[keep]
    nut = pd.read_csv(FNDDS / "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"])
    nut = nut[nut["nutrient_id"].isin(NUTRIENT_IDS)].pivot_table(index="fdc_id", columns="nutrient_id", values="amount")
    nut = nut.rename(columns=NUTRIENT_IDS)
    por = pd.read_csv(FNDDS / "food_portion.csv", usecols=["fdc_id", "portion_description", "gram_weight"])
    qns = por[por["portion_description"] == "Quantity not specified"].groupby("fdc_id")["gram_weight"].first()
    named = por[por["portion_description"] != "Quantity not specified"]

    def words(fid, grams):
        """The named portion closest in weight to the typical one, e.g. '1 small/regular'."""
        p = named[named["fdc_id"] == fid]
        return p.iloc[(p["gram_weight"] - grams).abs().argmin()]["portion_description"] if len(p) else f"{grams:.0f} g"

    rows = []
    for _, f in food.iterrows():
        fid = f["fdc_id"]
        if fid not in nut.index or not qns.get(fid, 0) > 0:
            continue
        grams = float(qns[fid])
        per100 = nut.loc[fid]
        rows.append({"fdc_id": fid, "food": f["description"], "category": f["category"], "grams": grams,
                     "portion": words(fid, grams),
                     **{n: round(float(np.nan_to_num(per100.get(n, 0))) * grams / 100, 1) for n in NUTRIENTS}})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE, index=False)
    return df


def prompt(name):
    """'Egg burrito, with bacon' -> 'a photo of egg burrito with bacon, a type of food'."""
    name = name.replace(", NFS", "").replace(",", "").lower()
    return f"a photo of {name}, a type of food"


# ---------- CLIP ----------

def unit(x):
    """Embedding as a unit vector. (transformers 5 returns an output object, older versions a tensor.)"""
    x = x if hasattr(x, "norm") else x.pooler_output
    return x / x.norm(dim=-1, keepdim=True)


@lru_cache(maxsize=1)
def foods_and_text():
    """The FNDDS table and CLIP's text vectors for every food (computed once, then read from disk: no model needed)."""
    foods = pd.read_csv(TABLE) if TABLE.exists() else build_table()
    if TEXT.exists() and len(np.load(TEXT)) == len(foods):
        return foods, np.load(TEXT)
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model, proc = CLIPModel.from_pretrained(MODEL).eval(), CLIPProcessor.from_pretrained(MODEL)
    out = []
    with torch.no_grad():
        for i in range(0, len(foods), 256):
            text = proc(text=[prompt(f) for f in foods["food"][i:i + 256]], return_tensors="pt", padding=True)
            out.append(unit(model.get_text_features(**text)).numpy())
    t = np.concatenate(out)
    CACHE.mkdir(exist_ok=True)
    np.save(TEXT, t)
    return foods, t


VISION = CACHE / "clip_vision_l14_fp16"


@lru_cache(maxsize=1)
def vision():
    """Only CLIP's image half (the text side is cached above), from a local half-precision copy (600 MB instead of
    1.7 GB: loads in about a second instead of 13), run in full precision."""
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from transformers.utils import logging
    logging.set_verbosity_error()
    if not VISION.exists():
        CLIPVisionModelWithProjection.from_pretrained(MODEL).half().save_pretrained(VISION)
        CLIPImageProcessor.from_pretrained(MODEL).save_pretrained(VISION)
    return (CLIPVisionModelWithProjection.from_pretrained(VISION, dtype=torch.float32).eval(),
            CLIPImageProcessor.from_pretrained(VISION))


def load():
    """Everything: image model, processor, FNDDS table, text vectors."""
    model, proc = vision()
    foods, t = foods_and_text()
    return model, proc, foods, t


OV_MODEL = CACHE / "clip_vision_l14_ov.xml"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def pixels(image):
    """CLIP's image preparation in NumPy: shortest side to 224 (bicubic), centre crop 224, scale, normalise."""
    from PIL import Image
    im = (Image.open(image) if not hasattr(image, "convert") else image).convert("RGB")
    w, h = im.size
    k = 224 / min(w, h)
    im = im.resize((max(224, int(w * k)), max(224, int(h * k))), Image.BICUBIC)        # long side rounded down, as CLIP does
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    a = np.asarray(im.crop((left, top, left + 224, top + 224)), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None]


@lru_cache(maxsize=1)
def fast_vision():
    """The image model through OpenVINO (Intel's open-source runtime) on the Intel GPU if there is one, else the CPU:
    about 0.13 s a photo on the GPU vs 1.5 s with PyTorch, and no PyTorch to import. None if unavailable."""
    if not OV_MODEL.exists():
        return None
    try:
        import openvino as ov
        core = ov.Core()
        core.set_property({"CACHE_DIR": str(CACHE / "ov_cache")})       # compiled once, then loaded in ~1 s
        model = core.read_model(str(OV_MODEL))
        model.reshape({0: [1, 3, 224, 224]})
        for device in ("GPU", "CPU"):
            if device in core.available_devices:
                try:
                    return core.compile_model(model, device), device
                except Exception:
                    continue
    except Exception:
        return None
    return None


def export_openvino():
    """Save the image model for OpenVINO (half-precision weights)."""
    import openvino as ov
    import torch
    model, _ = vision()

    class Image(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).image_embeds

    ov.save_model(ov.convert_model(Image(model), example_input=torch.zeros(1, 3, 224, 224)), str(OV_MODEL), compress_to_fp16=True)


def embed(images, batch=16):
    """CLIP image vectors (unit length). images: PIL images or file paths. OpenVINO when available, else PyTorch."""
    fast = fast_vision()
    if fast is not None:
        out = np.concatenate([fast[0](pixels(x))[0] for x in images])
        return out / np.linalg.norm(out, axis=1, keepdims=True)
    import torch
    from PIL import Image
    model, proc = vision()
    out = []
    for i in range(0, len(images), batch):
        ims = [Image.open(x).convert("RGB") if not hasattr(x, "convert") else x.convert("RGB") for x in images[i:i + batch]]
        with torch.no_grad():
            e = model(**proc(images=ims, return_tensors="pt")).image_embeds
        out.append((e / e.norm(dim=-1, keepdim=True)).numpy())
    return np.concatenate(out)


# ---------- Estimating ----------

def lookup(vectors, temperature, top):
    """Zero-shot over FNDDS: probability-weighted nutrition of the top matches at their typical portion.
    Returns (nutrition (n, 5), top indices (n, top), their weights (n, top))."""
    foods, text = foods_and_text()
    logits = temperature * vectors @ text.T
    best = np.argsort(-logits, axis=1)[:, :top]
    w = np.exp(np.take_along_axis(logits, best, axis=1) - np.take_along_axis(logits, best[:, :1], axis=1))
    w /= w.sum(axis=1, keepdims=True)
    nut = np.stack([(w * foods[n].to_numpy()[best]).sum(axis=1) for n in NUTRIENTS], axis=1)
    return nut, best, w


def recall(vectors, memory, k=5):
    """Nearest logged meal photos. Returns (nutrition (n, 5), best similarity (n,))."""
    sim = vectors @ memory["vectors"].T
    near = np.argsort(-sim, axis=1)[:, :k]
    s = np.take_along_axis(sim, near, axis=1)
    w = np.exp(20 * (s - s[:, :1]))
    w /= w.sum(axis=1, keepdims=True)
    return np.einsum("ik,ikn->in", w, np.nan_to_num(memory["nutrition"][near])), s[:, 0]


def blend(table_nut, mem_nut, sim, settings):
    """Memory weight rises from 0 to 1 as the best memory match gets closer (a logistic curve)."""
    scale = np.array([settings["scale"][n] for n in NUTRIENTS])
    if mem_nut is None:
        return table_nut * scale, np.zeros(len(table_nut))
    w = 1 / (1 + np.exp(-(sim - settings["memory_center"]) / settings["memory_width"]))
    return w[:, None] * mem_nut + (1 - w[:, None]) * table_nut * scale, w


@lru_cache(maxsize=1)
def settings():
    return {**DEFAULTS, **json.loads(SETTINGS.read_text())} if SETTINGS.exists() else DEFAULTS


@lru_cache(maxsize=1)
def load_memory():
    return dict(np.load(MEMORY)) if MEMORY.exists() else None


def estimate(image, portion=1.0):
    """One photo -> {'nutrition', 'items' [(food, portion words, grams, share)], 'memory' weight, 'detail'}.
    portion scales everything (1 = the usual amount)."""
    st = settings()
    v = embed([image])
    table_nut, best, w = lookup(v, st["temperature"], st["top"])
    memory = load_memory()
    mem_nut, sim = recall(v, memory) if memory is not None else (None, np.zeros(1))
    nut, mw = blend(table_nut, mem_nut, sim, st)
    _, _, foods, _ = load()
    items = [(foods["food"].iloc[i], foods["portion"].iloc[i], float(foods["grams"].iloc[i]), float(s))
             for i, s in zip(best[0], w[0])]
    detail = f"looks most like {items[0][0].lower()} ({items[0][3]:.0%} sure), usually {items[0][1]} ({items[0][2]:.0f} g)"
    if mw[0] > 0.5:
        detail += f"; also very close to meals logged before (similarity {sim[0]:.2f}), which count for {mw[0]:.0%}"
    return {"nutrition": dict(zip(NUTRIENTS, portion * nut[0])), "items": items, "memory": float(mw[0]), "detail": detail}


# ---------- Picking foods on the site ----------

def base_name(food):
    """'Burrito bowl, beef or pork' -> 'burrito bowl': variants of one dish share it."""
    return food.split(",")[0].strip().lower()


_ANALYSED = {}


def image_key(image):
    """Same photo -> same key (file path + time, or the pixels), so a photo is only looked at once."""
    import hashlib
    from pathlib import Path
    if isinstance(image, (str, Path)):
        return f"{image}:{Path(image).stat().st_mtime}"
    return hashlib.sha1(image.convert("RGB").tobytes()).hexdigest()


def analyse(image, distinct=3):
    """Cached version of _analyse (by photo)."""
    k = image_key(image)
    if k not in _ANALYSED:
        _ANALYSED[k] = _analyse(image, distinct)
    return _ANALYSED[k]


def _analyse(image, distinct=3):
    """A photo -> the most likely distinct dishes and what the photo memory says.
    Returns {'candidates': [(food index, share)], 'memory': nutrition dict or None, 'memory_weight', 'similarity'}.
    Shares are CLIP's confidence, summed over each dish's variants; each dish is shown by its likeliest variant."""
    return analyse_vector(embed([image]), distinct)


def analyse_vector(v, distinct=3):
    """_analyse from a CLIP image vector (1, 768) instead of a photo: the browser version computes the vector itself
    (transformers.js) since PyTorch doesn't run there."""
    v = np.asarray(v, dtype=np.float32).reshape(1, -1)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    st = settings()
    _, best, w = lookup(v, st["temperature"], 50)
    foods, _ = foods_and_text()
    dishes = {}
    for i, share in zip(best[0], w[0]):
        b = base_name(foods["food"].iloc[i])
        if b not in dishes:
            dishes[b] = [int(i), 0.0]
        dishes[b][1] += float(share)
    cands = sorted(dishes.values(), key=lambda x: -x[1])[:distinct]
    memory = load_memory()
    if memory is None:
        return {"candidates": cands, "memory": None, "memory_weight": 0.0, "similarity": 0.0}
    mem_nut, sim = recall(v, memory)
    weight = float(1 / (1 + np.exp(-(sim[0] - st["memory_center"]) / st["memory_width"])))
    near = np.argsort(-(v @ memory["vectors"].T)[0])[:5]
    same = lambda i: int((np.nan_to_num(memory["nutrition"]) == np.nan_to_num(memory["nutrition"][i])).all(axis=1).sum())
    neighbours = [{"photo": str(memory["photo"][i]) if "photo" in memory else None, "times_logged": same(i),
                   "subject": int(memory["subject"][i]) if "subject" in memory else None,
                   "similarity": float(v[0] @ memory["vectors"][i]),
                   **dict(zip(NUTRIENTS, np.nan_to_num(memory["nutrition"][i]).round(1).tolist()))} for i in near]
    return {"candidates": cands, "memory": dict(zip(NUTRIENTS, mem_nut[0].round(1).tolist())), "memory_weight": weight,
            "similarity": float(sim[0]), "neighbours": neighbours}


def food(index):
    """One FNDDS food: name, portion in words, grams and nutrition at its typical portion."""
    foods, _ = foods_and_text()
    r = foods.iloc[index]
    return {"name": r["food"], "portion": r["portion"], "grams": float(r["grams"]), **{n: float(r[n]) for n in NUTRIENTS}}


def plate(items, memory=None, memory_weight=0.0):
    """Nutrition of a plate: the sum of its foods (index, portion multiplier), blended with the photo memory by its
    weight (the tuned curve). Several foods eaten together are one meal at one time."""
    scale = np.array([settings()["scale"][n] for n in NUTRIENTS])
    total = np.zeros(len(NUTRIENTS))
    for index, portion in items:
        f = food(index)
        total += portion * np.array([f[n] for n in NUTRIENTS])
    total *= scale
    if memory is not None and memory_weight > 0:
        total = memory_weight * np.array([memory[n] for n in NUTRIENTS]) + (1 - memory_weight) * total
    return dict(zip(NUTRIENTS, total.round(1).tolist()))


def food_names():
    foods, _ = foods_and_text()
    return foods["food"].tolist()


# ---------- Building, tuning, testing ----------

def cgmacros_photos():
    """Every CGMacros meal photo with its true nutrition and study day."""
    import metabolic
    rows = []
    for subject in metabolic.ids():
        folder = metabolic.CGMACROS / f"CGMacros-{subject:03d}"
        raw = pd.read_csv(folder / f"CGMacros-{subject:03d}.csv", parse_dates=["Timestamp"])
        raw.columns = raw.columns.str.strip()
        first = raw["Timestamp"].min().normalize()
        for _, m in raw[raw["Meal Type"].notna() & raw["Image path"].notna()].iterrows():
            if (folder / m["Image path"]).exists():
                rows.append({"subject": subject, "day": (m["Timestamp"].normalize() - first).days + 1,
                             "photo": str(folder / m["Image path"]), **{n: m[n.title()] for n in NUTRIENTS}})
    return pd.DataFrame(rows)


def mae(est, truth):
    return np.nanmean(np.abs(est - truth), axis=0)


def tune(V, truth, mem_mask, eval_mask):
    """Pick temperature, top-k, memory curve and lookup scale on (memory from mem_mask) -> eval_mask photos,
    minimising the carbs error (what the glucose models care about most), calories as a tie-breaker."""
    memory = {"vectors": V[mem_mask], "nutrition": truth[mem_mask]}
    mem_nut, sim = recall(V[eval_mask], memory)
    best = None
    for temperature in (50.0, 100.0):
        for top in (1, 5, 20):
            tab, _, _ = lookup(V[eval_mask], temperature, top)
            ratio = np.nanmedian(truth[eval_mask], axis=0) / np.maximum(np.nanmedian(tab, axis=0), 1)
            for scale in (np.ones(5), ratio):
                for center in (-1.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 2.0):     # -1: always memory, 2: never
                    for width in (0.02, 0.05):
                        s = {"temperature": temperature, "top": top, "memory_center": center, "memory_width": width,
                             "scale": dict(zip(NUTRIENTS, np.round(scale, 3).tolist()))}
                        e = mae(blend(tab, mem_nut, sim, s)[0], truth[eval_mask])
                        key = (e[0], e[4])
                        if best is None or key < best[0]:
                            best = (key, s)
    return best[1]


def main():
    print("FNDDS table...", flush=True)
    foods = build_table()
    print(f"{len(foods)} foods with a typical portion", flush=True)
    load()
    meals = cgmacros_photos()
    if PHOTOS.exists() and len(np.load(PHOTOS)["V"]) == len(meals):
        V = np.load(PHOTOS)["V"]
    else:
        print(f"embedding {len(meals)} meal photos with {MODEL}...", flush=True)
        V = embed(list(meals["photo"]))
        np.savez(PHOTOS, V=V)
    truth = meals[NUTRIENTS].to_numpy(float)
    day = meals["day"].to_numpy()
    if not OV_MODEL.exists():
        export_openvino()
    s = tune(V, truth, day <= 5, (day >= 6) & (day <= 7))
    SETTINGS.write_text(json.dumps(s, indent=2))
    print("settings (tuned on days 1-5 -> 6-7):", s, flush=True)

    train, test = day <= 7, day >= 8
    memory = {"vectors": V[train], "nutrition": truth[train], "photo": meals.loc[train, "photo"].to_numpy(str),
              "subject": meals.loc[train, "subject"].to_numpy(int)}
    np.savez(MEMORY, **memory)
    tab, best, _ = lookup(V[test], s["temperature"], s["top"])
    mem_nut, sim = recall(V[test], memory)
    combined, mw = blend(tab, mem_nut, sim, s)
    median = np.nanmedian(truth[train], axis=0)
    rows = {"always the median meal": mae(np.tile(median, (test.sum(), 1)), truth[test]),
            "USDA lookup only": mae(blend(tab, None, sim, s)[0], truth[test]),
            "photo memory only": mae(mem_nut, truth[test]), "combined": mae(combined, truth[test])}
    report = pd.DataFrame(rows, index=NUTRIENTS).T.round(1)
    print(f"Test: {test.sum()} photos from day 8+ (mean absolute error):\n{report.to_string()}", flush=True)
    report.to_csv(ROOT / "results" / "photo2nutrition_scores.csv")
    _, _, foods, _ = load()
    out = meals[test].drop(columns="photo").assign(top_food=foods["food"].to_numpy()[best[:, 0]], memory_weight=mw.round(2),
                                                   **{f"est_{n}": combined[:, j] for j, n in enumerate(NUTRIENTS)})
    out.round(1).to_csv(ROOT / "results" / "photo2nutrition.csv", index=False)


if __name__ == "__main__":
    main()
