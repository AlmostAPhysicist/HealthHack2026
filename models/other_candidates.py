"""Random people as drag-and-drop sets, to see how the site does on ordinary cases (not shown on the site).

10 people sampled at random from everyone: 5 without type 1 (CGMacros) and 5 with type 1 (MetaboNet), never-seen and
trained-on alike. For each, a random meal and a random whole day after day 7 (not the best predicted ones), in the
same files as samples/try_me/: meal, meal_check, day, day_check.

Never-seen or not: the site takes CGMacros people out of the training layers when they are uploaded (their about-me
file names them), so they are never-seen there. Type 1 training people stay in the model, so their results look
better than a new person's would; their folder says "trained on".

Run:  python models/other_candidates.py   -> samples/test_people/
"""
import shutil

import numpy as np
import pandas as pd

import metabolic
import t1d
from common import ROOT
from demo_files import build, photo_for

OUT = ROOT / "samples" / "test_people"
rng = np.random.default_rng(11)


def random_meal(series, backend, pid):
    """A meal after day 7 with 15 g carbs or more and 2 h of glucose after it (with a photo for CGMacros if possible)."""
    first = series.index[0].normalize() + pd.Timedelta(days=7)
    g, c = series["glucose_raw"].to_numpy(float), series["carbs"].to_numpy(float)
    ok = [a for a in np.flatnonzero(c >= 15) if series.index[a] >= first and a + 24 < len(g)
          and np.isfinite(g[a: a + 25]).sum() >= 20 and np.isfinite(g[a])]
    if backend == "metabolic":
        with_photo = [a for a in ok if photo_for(pid, series.index[a]) is not None]
        ok = with_photo or ok
    return str(series.index[rng.choice(ok)]) if ok else None


def random_day(series):
    """A day after day 7 with 90 % of its glucose and at least 2 meals."""
    first = series.index[0].normalize()
    days = [d0 for d0 in pd.date_range(first + pd.Timedelta(days=7), series.index[-1].normalize() - pd.Timedelta(days=1), freq="D")
            if (lambda d: d["glucose_raw"].notna().mean() >= 0.9 and (d["carbs"] > 0).sum() >= 2)(
                series[(series.index >= d0) & (series.index < d0 + pd.Timedelta(days=1))])]
    return str(rng.choice(days).date()) if days else None


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    picked = []
    for backend, module, n in (("metabolic", metabolic, 5), ("t1d", t1d, 5)):
        ids = metabolic.ids() if backend == "metabolic" else [i["id"] for _, i in t1d.load_all()]
        for pid in rng.permutation(ids):
            if sum(b == backend for b, _ in picked) == n:
                break
            key = int(pid) if backend == "metabolic" else pid
            series, info = module.load(key)
            meal, day = random_meal(series, backend, key), random_day(series)
            if meal is None or day is None:
                continue
            seen = "never seen" if backend == "metabolic" or t1d.is_test(info["id"]) else "trained on"
            build(f"{info['id']}, {seen}", "try_me", backend, key, meal, day, out=OUT)
            picked.append((backend, info["id"]))
    print("built", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
