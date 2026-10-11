"""Choose the demo and sample people from pick_demos.py's results: the people the current model predicts best.

Every never-seen person's meals were predicted (results/demo_candidates.csv, the model the site uses). People are
ranked by their median meal error (how well the model knows them overall, not one lucky meal); metabolic people need a
meal photo. Then pick_days.py and demo_files.py use results/demo_people.json.

    Home demo:  Maya (best healthy or prediabetic), Sam (best type 2), Ana (best type 1)
    Samples:    Jo (next best metabolic), Theo (next best type 1)
Names are just labels for anonymous study participants.

Run after pick_demos.py:  python models/choose_demos.py
"""
import json

import pandas as pd

from common import ROOT
from demo_files import photo_for


def main():
    c = pd.read_csv(ROOT / "results" / "demo_candidates.csv", parse_dates=["time"])
    c = c[(c["true_rise"] >= 30) & (c["carbs"] >= 20)]
    meta = c["backend"] == "metabolic"
    has_photo = [m and photo_for(int(i.split("-")[1]), t) is not None for m, i, t in zip(meta, c["id"], c["time"])]
    c = c[~meta | pd.Series(has_photo, index=c.index)]
    people = (c.groupby(["backend", "id", "group"]).agg(median_rmse=("rmse", "median"), best=("rmse", "min"), meals=("rmse", "size"))
              .reset_index())
    people = people[people["meals"] >= 3].sort_values("median_rmse")
    print(people.head(15).to_string(index=False))
    pick = lambda q, skip=(): people[q & ~people["id"].isin(skip)].iloc[0]["id"]
    m = people["backend"] == "metabolic"
    maya = pick(m & people["group"].isin(["healthy", "prediabetes"]))
    sam = pick(m & (people["group"] == "t2d"))
    ana = pick(people["backend"] == "t1d")
    jo = pick(m, (maya, sam))
    theo = pick(people["backend"] == "t1d", (ana,))
    out = {"Maya": ["demo", "metabolic", maya], "Sam": ["demo", "metabolic", sam], "Ana": ["demo", "t1d", ana],
           "Jo": ["try_me", "metabolic", jo], "Theo": ["try_me", "t1d", theo]}
    (ROOT / "results" / "demo_people.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
