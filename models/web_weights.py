"""A slim copy of the weights for the in-browser site (GitHub Pages): models_web.pkl.

Each model keeps every person's statistics so it can take a person out of the priors when their file names them
(never-seen). In the browser that is only needed for the people we ship (demo, sample and test people) and for
CGMacros people (small), so the type 1 people and the overlay's per-person statistics are dropped otherwise.
Predictions are identical; only the download shrinks (57 MB -> a few MB).

Run:  python models/web_weights.py
"""
import pickle
from pathlib import Path

import pandas as pd

from common import ROOT, WEIGHTS


def shipped_ids():
    ids = set()
    for f in list((ROOT / "samples").rglob("*about_me*.csv")) + list((ROOT / "static" / "try_me").rglob("*about_me*.csv")):
        row = pd.read_csv(f)
        if "dataset_id" in row:
            ids.update(str(v) for v in row["dataset_id"].dropna())
    return ids


def main():
    keep = shipped_ids()
    with open(WEIGHTS / "models.pkl", "rb") as f:
        bundle = pickle.load(f)
    for (backend, kind), model in bundle.items():
        stats = getattr(model, "person_stats", {})
        model.person_stats = {k: v for k, v in stats.items()
                              if k in keep or (backend == "metabolic" and kind == "regressor")}
    out = WEIGHTS / "models_web.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"kept {len(keep)} shipped people; {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
