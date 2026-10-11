"""Compare model versions on the same never-seen-patient test (learn days 1-7, predict day 8+, raw CGM).

    v1  meal inputs only, no cleaning                       (first version)
    v2  + jitter cleaning + meal number / day-so-far / exercise / pre-meal glucose / own past spikes
    v3  + calories, clock time, time since last meal, usual level, swings, large-meal saturation,
        and (overlay) a per-meal time-warp
    v4  + non-meal events: exercise (after a meal / on its own) and quiet time-of-day blocks
Each version is tuned on its own (days 1-5 -> 6-7), so the comparison is fair.
For speed, type 1 versions use all 48 test patients + 100 of the training people for the priors
(the final models in train.py use everyone).

Extra for type 1: OhioT1DM test patients with priors from only the other Ohio people (the old setup)
vs from the ~334-person training cohort.

Run:  python models/compare.py      -> results/versions.csv, results/versions.png
"""
import os
os.environ["MKL_THREADING_LAYER"] = "TBB"   # multi-threaded; MKL's default (Intel OpenMP) crashes in this env

import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metabolic
import t1d
from common import ROOT, split_days, baseline
from overlay import Overlay
from regressor import Regressor
from train import SHAPE_DAYS, SHAPE_EVALS, SHAPE_PEOPLE, Split, held_out_rule, summarize, tune

warnings.filterwarnings("ignore")
RESULTS = ROOT / "results"
T0 = time.time()


def versions(backend):
    """(name, regressor inputs, overlay events, clean glucose?, time-warp?)"""
    return [("v1", backend.REGRESSOR_V1, backend.OVERLAY_V1, False, False),
            ("v2", backend.REGRESSOR_V2, backend.OVERLAY_V2, True, False),
            ("v3", backend.REGRESSOR_V3, backend.OVERLAY_V3, True, True),
            ("v4", backend.REGRESSOR_INPUTS, backend.OVERLAY_V4, True, True)]


def evaluate(people, held, reg_inputs, ov_events, warp, only=None):
    """Tune + test both kinds. `only` = keep only these test ids in the scores. Returns {kind: summary row}."""
    train_people = [(s, i) for s, i in people if held is None or not held(i["id"])]
    shape_data = [(split_days(s, 7 if held is None else SHAPE_DAYS)[0], i) for s, i in train_people][:SHAPE_PEOPLE]
    out = {}
    for kind in ("regressor", "overlay"):
        model = Regressor(reg_inputs) if kind == "regressor" else Overlay(ov_events, warp=warp).fit_shapes(shape_data, SHAPE_EVALS, log=lambda m: None)
        full = {i["id"]: model.stats(s, baseline(s)) for s, i in train_people} if held else None
        setting, _ = tune(Split(model, people, held, 5, 2, full))
        res = Split(model, people, held, 7, prior_stats=full).score(*setting)
        if only is not None:
            res = res[res["id"].isin(only)]
        out[kind] = summarize(res).loc["person"]
    return out


def main():
    rows = []
    for backend in (metabolic, t1d):
        people = backend.load_all()
        held = held_out_rule(backend)
        if held is not None:                       # all test patients + 100 training people
            people = [p for p in people if held(p[1]["id"])] + [p for p in people if not held(p[1]["id"])][:100]
        for name, reg, ov, cleaned, warp in versions(backend):
            ppl = people if cleaned else [(s.assign(glucose=s["glucose_raw"]), i) for s, i in people]
            for kind, row in evaluate(ppl, held, reg, ov, warp).items():
                rows.append({"backend": backend.NAME, "kind": kind, "version": name, **row.to_dict()})
                print(f"[{(time.time() - T0) / 60:5.1f} min] {backend.NAME} {kind} {name}: RMSE {row['rmse_all']:.1f}, "
                      f"top-spike bias {row['top_spike_bias']:+.1f}", flush=True)
        if backend is t1d:                          # does training on hundreds of people help?
            ohio = [(s, i) for s, i in people if i["id"].startswith("OhioT1DM")]
            ohio_ids = [i["id"] for _, i in ohio]
            for setup, ppl, rule, only in [("v4, priors from 11 other Ohio people", ohio, None, None),
                                           ("v4, priors from 100 training people", people, held, ohio_ids)]:
                for kind, row in evaluate(ppl, rule, backend.REGRESSOR_INPUTS, backend.OVERLAY_V4, True, only).items():
                    rows.append({"backend": "t1d (Ohio test patients)", "kind": kind, "version": setup, **row.to_dict()})
                    print(f"[{(time.time() - T0) / 60:5.1f} min] Ohio {kind} {setup}: RMSE {row['rmse_all']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "versions.csv", index=False)
    plot(df[df["version"].isin(["v1", "v2", "v3", "v4"])], RESULTS / "versions.png")


def plot(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")
    colors = {"v1": "#c3c2b7", "v2": "#86b6ef", "v3": "#2a78d6", "v4": "#0d366b"}
    for ax, metric, title in [(axes[0], "rmse_all", "RMSE 30-120 min after meals (mg/dL)"),
                              (axes[1], "top_spike_mae", "Error on each day's biggest spike (mg/dL)")]:
        ax.set_facecolor("#fcfcfb")
        keys = list(df.groupby(["backend", "kind"], sort=False).groups)
        x = np.arange(len(keys))
        for j, v in enumerate(["v1", "v2", "v3", "v4"]):
            vals = [df[(df.backend == b) & (df.kind == k) & (df.version == v)][metric].iloc[0] for b, k in keys]
            bars = ax.bar(x + (j - 1.5) * 0.2, vals, 0.19, color=colors[v], label=v)
            ax.bar_label(bars, fmt="%.1f", fontsize=7, color="#52514e")
        ax.set_xticks(x, [f"{b}\n{k}" for b, k in keys], fontsize=8)
        ax.set_title(title, fontsize=10)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#898781", alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
