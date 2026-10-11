"""Save each Home demo person's track record (their last 20 meals replayed: the misses behind "typically off by"
and their personal band) next to their files, so the site doesn't replay them for every visitor. That replay is the
slowest part of a prediction, and in the browser (GitHub Pages) it takes half a minute. Rerun after retraining.

Run:  python models/demo_tracks.py   -> samples/home_demo/<person>/track_record.npz
"""
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import pandas as pd

from common import ROOT

sys.path.insert(0, str(ROOT / "views"))
import engine  # noqa: E402

DEMO = ROOT / "samples" / "home_demo"


def main():
    for folder in sorted(p for p in DEMO.iterdir() if p.is_dir()):
        ab = pd.read_csv(folder / "about_me.csv").iloc[0]
        history = engine.read(folder / "history.csv")
        info = {"group": ab["diabetes_type"], **({"a1c": ab["a1c"]} if pd.notna(ab.get("a1c")) else {})}
        person = engine.learn(history, info, f"demo-{folder.name}", exclude=ab.get("dataset_id"))
        track = engine.track_record(person, history)
        if track:
            np.savez(folder / "track_record.npz", resid=track["resid"], mae=track["mae"], rmse=track["rmse"],
                     meals=track["meals"])
            print(folder.name, f"rmse {track['rmse']:.1f} over {track['meals']} meals")


if __name__ == "__main__":
    main()
