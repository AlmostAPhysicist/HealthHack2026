"""Population misses for the personal band (views/engine.personal_band). Run after train.py.

For each backend and model kind: replay never-seen people's days 2-7 meal by meal (learn from everything before each
meal, predict the 3 h after it) and store the misses (truth - prediction, 36 steps) on the saved model as pop_resid.
The site pools a person's own replayed misses with 10 of these, so a short history can't make their band
overconfident. Never-seen: metabolic people are each taken out of the priors; type 1 uses the held-out test people
(their days 2-7 only, never the test days).

Run:  python models/band_pop.py
"""
import os
os.environ["MKL_THREADING_LAYER"] = "TBB"

import pickle
import warnings

import numpy as np
import pandas as pd

import metabolic
import t1d
from common import WEIGHTS, baseline, split_days

warnings.filterwarnings("ignore")
STEPS, PER_PERSON = 36, 20


def replay(model, series, info):
    hist = split_days(series, 7)[0]
    base = baseline(hist)
    prior = model.prior(info["group"], exclude=info["id"] if info["id"] in getattr(model, "person_stats", {}) else None)
    g, raw = hist["glucose"].to_numpy(float), hist["glucose_raw"].to_numpy(float)
    first = len(split_days(series, 1)[0])
    meals = [a for a in np.flatnonzero(hist["carbs"].to_numpy() > 0)
             if a >= first and a + 24 < len(g) and np.isfinite(g[a]) and np.isfinite(raw[a + 1: a + 25]).sum() >= 20][-PER_PERSON:]
    rows, out = model.rows(hist, base), []
    for a in meals:
        coef = model.personalize(hist, info["group"], base, until=a, prior=prior, rows=rows)
        p = model.predict_from(hist, a, coef, base)[0][:STEPS]
        r = np.full(STEPS, np.nan)
        stop = min(a + 1 + STEPS, len(raw))
        r[: stop - a - 1] = raw[a + 1: stop] - p[: stop - a - 1]
        out.append(r)
    return out


def main():
    with open(WEIGHTS / "models.pkl", "rb") as f:
        bundle = pickle.load(f)
    for backend in (metabolic, t1d):
        people = backend.load_all()
        if backend is t1d:
            people = [p for p in people if t1d.is_test(p[1]["id"])]
        for (b, kind), model in bundle.items():
            if b != backend.NAME:
                continue
            res = [r for s, i in people for r in replay(model, s, i)]
            model.pop_resid = np.array(res, dtype=np.float32)
            q = np.nanquantile(model.pop_resid, [0.1, 0.9], axis=0)
            print(f"{b} {kind}: {len(res)} replayed meals; 80 % of misses at 60 min between {q[0][11]:.0f} and +{q[1][11]:.0f}",
                  flush=True)
    with open(WEIGHTS / "models.pkl", "wb") as f:
        pickle.dump(bundle, f)


if __name__ == "__main__":
    main()
