"""Build showcase/: a few real patients from both datasets, with both models' day-8 predictions vs the truth.

Per patient (showcase/<id>/):
    history.csv       days 1-7 (website upload format)
    day8_meals.csv    day 8 meals in the website's meal-CSV format (time, meal, carbs, protein, fat, fiber, calories, insulin)
    day8_exercise.csv day 8 exercise sessions detected from activity (website's exercise-CSV format)
    day8_inputs.csv   day 8 recorded inputs: meals / insulin / activity / HR, no glucose
    validation.csv    day 8 real glucose (hidden from the model; pick it in the website to score a prediction)
    info.json         what we know about the patient + how the classifier routed them
    predictions.csv   both models: whole-day forecast and per-meal forecasts, next to the real glucose
    plot.png          the same, as a picture
showcase/summary.csv  error per patient and model

Every showcase patient is never-seen: type 1 picks come from the held-out test people (priors from the
training people only); metabolic picks are taken out of the priors (fit on everyone's days 1-7). The model
then learns the patient from their days 1-7 only, so day 8 is a fair test.
Run after train.py:  python models/showcase.py
"""
import os
os.environ["MKL_THREADING_LAYER"] = "TBB"

import json
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metabolic
import t1d
from classifier import classify, load_classifier
from common import ROOT, WEIGHTS, MEAL_STEPS, split_days, baseline, predict_window
from events import exercise_table
from train import held_out_rule, make

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = ROOT / "showcase"
PICK_BY = {"metabolic": "group", "t1d": "study"}   # typical people per diabetes group / per t1d study
PER_SET = {"healthy": 2, "prediabetes": 2, "t2d": 2, "OhioT1DM": 2, "Loop": 2, "PEDAP": 1, "AZT1D": 1, "HUPA-UCM": 1}
COLORS = {"actual": "#2a78d6", "regressor": "#eb6834", "overlay": "#1baf7a"}
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"


def typical_people(backend):
    """Per diabetes group (metabolic) or study (t1d): the person whose test error (regressor) is closest to
    that set's median. Not cherry-picked."""
    err = pd.read_csv(ROOT / "results" / "test_errors.csv")
    err = err[(err["backend"] == backend) & (err["kind"] == "regressor")]
    e = err.groupby(["group", "id"])[["err30", "err60", "err90", "err120"]].apply(
        lambda d: np.sqrt(np.mean(d.to_numpy() ** 2))).rename("rmse").reset_index()
    e["study"] = e["id"].str.rsplit("-", n=1).str[0]
    picks = []
    for name, g in e.groupby(PICK_BY[backend]):
        picks += g.iloc[(g["rmse"] - g["rmse"].median()).abs().argsort()[:PER_SET.get(name, 0)]]["id"].tolist()
    return picks


def day8_predictions(models, coefs, series, info):
    """Both models on day 8. 'day' = forecast from the end of day 7 (no day-8 CGM at all).
    'meal' = from each meal, knowing glucose up to that meal (the scored setup)."""
    hist, future = split_days(series)
    day8 = future[future.index < future.index[0] + pd.Timedelta(days=1)]
    full = pd.concat([hist, day8])
    blind = full.copy()
    blind.loc[day8.index, "glucose"] = np.nan
    base = baseline(hist)
    out = pd.DataFrame({"actual": day8["glucose_raw"]}, index=day8.index)             # truth = raw CGM
    meals = full.index.get_indexer(day8.index[day8["carbs"] > 0])
    g = full["glucose"].to_numpy()
    for kind, model in models.items():
        coef = coefs[kind]
        out[f"{kind}_day"], _ = predict_window(model, blind, coef, day8.index[0], day8.index[-1], base)
        out[f"{kind}_meal"] = np.nan
        out[f"{kind}_meal_lo"] = out[f"{kind}_meal_hi"] = np.nan
        for a in meals:
            if np.isnan(g[a]) or a + 24 >= len(full):
                continue
            mean, sd = model.predict_with_sd(full, a, coef, base)          # coef = (coef, cov, s2)
            pred, band = mean[0][:24], 1.2816 * sd[0][:24]
            t = full.index[a + 1: a + 25]
            out.loc[t, f"{kind}_meal"] = pred
            out.loc[t, f"{kind}_meal_lo"] = pred - band
            out.loc[t, f"{kind}_meal_hi"] = pred + band
    return out, day8


def meal_errors(out, kind):
    """RMSE at 30/60/90/120 min after each day-8 meal."""
    meal_times = out.index[out[f"{kind}_meal"].notna() & out[f"{kind}_meal"].shift(1).isna()] - pd.Timedelta(minutes=5)
    errs = []
    for t0 in meal_times:
        for m in MEAL_STEPS:
            t = t0 + pd.Timedelta(minutes=5 * m)
            if t in out.index:
                errs.append(out.loc[t, f"{kind}_meal"] - out.loc[t, "actual"])
    return float(np.sqrt(np.nanmean(np.square(errs)))) if errs else np.nan


def plot(out, day8, info, route, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, facecolor=SURFACE)
    for ax, mode, title in [(axes[0], "day", "Whole day 8 forecast from the end of day 7 (no day-8 CGM)"),
                            (axes[1], "meal", "Per-meal forecasts (glucose known up to each meal), 80% band")]:
        ax.set_facecolor(SURFACE)
        ax.plot(out.index, out["actual"], color=COLORS["actual"], lw=2, label="Actual (validation)")
        for kind in ("regressor", "overlay"):
            ax.plot(out.index, out[f"{kind}_{mode}"], color=COLORS[kind], lw=2, ls="--", label=f"{kind} model")
            if mode == "meal":
                ax.fill_between(out.index, out[f"{kind}_meal_lo"], out[f"{kind}_meal_hi"], color=COLORS[kind], alpha=0.1, lw=0)
        for t in day8.index[day8["carbs"] > 0]:
            ax.axvline(t, color=MUTED, lw=0.8, ls=":")
            ax.text(t, 0.98, f" {day8.loc[t, 'carbs']:.0f}g", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8, va="top")
        for t in day8.index[day8["bolus"] > 0.5]:
            ax.plot(t, 0.02, marker="^", color=MUTED, transform=ax.get_xaxis_transform(), ms=6)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_ylabel("Glucose (mg/dL)", color=INK)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=MUTED, alpha=0.15, lw=0.5)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"{info['id']}: {info['group']} -> {route['backend']} backend ({route['reason']}). "
                 f"Dotted lines = meals (g carbs), triangles = insulin doses", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def export(series, info, route, out, folder):
    """Write the patient in the website's upload format, plus predictions."""
    folder.mkdir(parents=True, exist_ok=True)
    series = series.assign(glucose=series["glucose_raw"]).drop(columns="glucose_raw")   # files hold raw CGM
    hist, future = split_days(series)
    day8 = future[future.index < future.index[0] + pd.Timedelta(days=1)]
    cols = [c for c in series.columns if c == "glucose" or series[c].fillna(0).abs().sum() > 0]
    hist[cols].rename_axis("time").reset_index().to_csv(folder / "history.csv", index=False)
    day8[[c for c in cols if c != "glucose"]].rename_axis("time").reset_index().to_csv(folder / "day8_inputs.csv", index=False)
    day8[["glucose"]].rename_axis("time").reset_index().to_csv(folder / "validation.csv", index=False)
    out.round(1).rename_axis("time").reset_index().to_csv(folder / "predictions.csv", index=False)
    # the day's meals and exercise in the website's CSV formats (for the meal / exercise uploaders)
    meals = day8[day8["carbs"] > 0]
    bolus = day8["bolus"]
    pd.DataFrame({
        "time": meals.index.strftime("%H:%M"),
        "meal": ["Breakfast" if 4 <= t.hour < 11 else "Lunch" if t.hour < 16 else "Dinner" if t.hour < 21 else "Snack" for t in meals.index],
        **{c: meals[c].round(1).to_numpy() for c in ["carbs", "protein", "fat", "fiber", "calories"]},
        "insulin": [round(float(bolus[(bolus.index >= t - pd.Timedelta(minutes=30)) & (bolus.index <= t + pd.Timedelta(minutes=30))].sum()), 2)
                    for t in meals.index]}).to_csv(folder / "day8_meals.csv", index=False)
    ex = exercise_table(pd.concat([hist, day8]).assign(glucose=np.nan))
    ex = ex[ex["step"] >= len(hist)]
    t0 = pd.concat([hist, day8]).index
    pd.DataFrame({"start": [t0[int(a)].strftime("%H:%M") for a in ex["step"]], "minutes": ex["minutes"].astype(int).to_numpy(),
                  "type": "Recorded", "intensity": ["hard" if z > 5 else "moderate" if z > 3 else "light" for z in ex["intensity"]]}
                 ).to_csv(folder / "day8_exercise.csv", index=False)
    meta = {k: (v if isinstance(v, str) else float(v)) for k, v in info.items() if pd.notna(v)}
    meta["route"] = route
    (folder / "info.json").write_text(json.dumps(meta, indent=2, default=str))


def main():
    import shutil
    for old in OUT.glob("*/info.json"):                 # replace the previous demo patients
        shutil.rmtree(old.parent)
    with open(WEIGHTS / "models.pkl", "rb") as f:
        bundle = pickle.load(f)
    clf = load_classifier()
    summary = []
    for backend in (metabolic, t1d):
        people = backend.load_all()
        held = held_out_rule(backend)
        picks = typical_people(backend.NAME)
        # priors that never saw the picks: t1d = training people (all days); metabolic = everyone's days 1-7
        prior_people = [(split_days(s)[0] if held is None else s, i) for s, i in people if held is None or not held(i["id"])]
        models = {}
        for kind in ("regressor", "overlay"):
            final = bundle[(backend.NAME, kind)]
            models[kind] = make(kind, backend, final if kind == "overlay" else None).fit_priors(prior_people)
            models[kind].set_strengths(*final.setting)
            models[kind].sigma, models[kind].kappa = final.sigma, final.kappa
        for series, info in people:
            if info["id"] not in picks:
                continue
            hist = split_days(series)[0]
            base = baseline(hist)
            coefs = {k: m.posterior(hist, info["group"], base,
                                    prior=m.prior(info["group"], exclude=info["id"] if info["id"] in m.person_stats else None))
                     for k, m in models.items()}
            # route the way the website would: only the history, no stated type, no labs
            route = classify({}, hist, clf)
            out, day8 = day8_predictions(models, coefs, series, info)
            folder = OUT / info["id"]
            export(series, info, route, out, folder)
            plot(out, day8, info, route, folder / "plot.png")
            row = {"id": info["id"], "group": info["group"], "routed_to": route["group"], "route_reason": route["reason"]}
            for kind in ("regressor", "overlay"):
                row[f"{kind}_meal_rmse"] = round(meal_errors(out, kind), 1)
                row[f"{kind}_day_rmse"] = round(float(np.sqrt(np.nanmean((out[f"{kind}_day"] - out["actual"]) ** 2))), 1)
            summary.append(row)
            print(row)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
