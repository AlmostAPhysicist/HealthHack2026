"""Train and test both model kinds (regressor, overlay) on both backends (metabolic, t1d). Save weights.

Every test patient is someone the model never saw:
    t1d        ~334 training people fit the priors (all 28 days each). 48 test people (all OhioT1DM + 10 % of
               the rest) never touch the priors.
    metabolic  45 people is too few to set some aside, so each test person is taken out of the population
               and group priors before they are scored.
Then the model learns the test person from their days 1-7 and predicts every meal after day 7
(30/60/90/120 min), scored against raw CGM.

Steps per backend and model kind:
1. Tune shrinkage (group, person, loose factor for person-specific features) on days 1-5 -> 6-7.
   Also calibrate the error bands there: the raw uncertainty (coefficient covariance + residual noise)
   is too narrow because neighbouring 5-min rows are correlated, so it is scaled per horizon until it
   matches the real errors on days 6-7. Test-day coverage of the 80 % band is then reported as is.
2. Test: learn days 1-7, predict day 8+.
3. Meal by meal: start knowing nothing about a person, add one meal at a time, predict the next.
4. Save final weights (fit on everyone) to models/weights/models.pkl.

Run:  python models/train.py               both backends
      python models/train.py metabolic     one backend only (keeps the other's saved weights and results)
Results go to results/.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "TBB"   # multi-threaded; MKL's default (Intel OpenMP) crashes in this env   # MKL's default threading crashes in this env

import json
import pickle
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metabolic
import t1d
from common import ROOT, WEIGHTS, HORIZON, MEAL_STEPS, Stats, solve, posterior, row_variance, split_days, baseline, meal_anchors
from overlay import Overlay
from regressor import V5, Regressor

warnings.filterwarnings("ignore", category=RuntimeWarning)
RESULTS = ROOT / "results"
BACKENDS = [metabolic, t1d]
KINDS = ["regressor", "overlay"]
ERRS = ["err30", "err60", "err90", "err120"]
GRID = [(gs, ps, lf) for gs in (10000.0, 30000.0, 100000.0, 300000.0, 1e6) for ps in (1000.0, 3000.0, 10000.0, 30000.0, 100000.0)
        for lf in (1.0, 3.0, 10.0, 30.0)]          # wide enough that the best settings land inside it, not on its edge
SHAPE_PEOPLE, SHAPE_DAYS, SHAPE_EVALS, FOOD_EVALS = 45, 14, 80, 250   # overlay shapes: 45 people x 14 days,
                                                                        # 250 optimizer steps for meals, 80 for the rest
COLORS = {("metabolic", "regressor"): "#2a78d6", ("metabolic", "overlay"): "#86b6ef",
          ("t1d", "regressor"): "#eb6834", ("t1d", "overlay"): "#f4b393"}
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def make(kind, backend, shapes=None):
    """New model of the given kind for a backend. `shapes` = an Overlay whose fitted shapes to reuse."""
    if kind == "regressor":
        return Regressor(backend.REGRESSOR_V7, extra=V5)
    model = Overlay(backend.OVERLAY_EVENTS)
    return model.copy_shapes(shapes) if shapes is not None else model


def held_out_rule(backend):
    """Who is a never-seen test person. None = everyone, each removed from the priors in turn."""
    return getattr(backend, "is_test", None)


def scored_meals(series, start=0):
    """Meals from step `start` on with a full 2 h of glucose after them. Also: did another meal land in those 2 h?"""
    g, carbs = series["glucose"].to_numpy(), series["carbs"].to_numpy()
    return [(a, carbs[a + 1: a + 25].sum() > 0) for a in meal_anchors(series)
            if a >= start and a + 24 < len(g) and not np.isnan(g[a: a + 25]).any()]


def rmse(df):
    return float(np.sqrt(np.mean(df[ERRS].to_numpy() ** 2)))


class Split:
    """One train/test split with everything that does not depend on the shrinkage settings precomputed,
    so many settings can be scored quickly."""

    def __init__(self, model, people, held_out, train_days=7, test_days=None, prior_stats=None):
        self.model, self.tests = model, []
        self.prior_stats, self.groups = {}, {}
        for series, info in people:
            pid = info["id"]
            test_person = held_out is None or held_out(pid)
            hist = split_days(series, train_days)[0]
            self.groups[pid] = info["group"]
            if held_out is None:                                    # priors from everyone's history
                self.prior_stats[pid] = model.stats(hist, baseline(hist))
            elif not test_person:                                   # priors from training people's full data
                self.prior_stats[pid] = (prior_stats or {}).get(pid) or model.stats(series, baseline(series))
            if not test_person:
                continue
            if test_days is not None:
                series = series[series.index < series.index[0].normalize() + pd.Timedelta(days=train_days + test_days)]
            meals = scored_meals(series, start=len(hist))
            if not meals:
                continue
            base = baseline(hist)
            anchors = np.array([a for a, _ in meals])
            X, _ = model.design(series, anchors, base)
            g, raw = series["glucose"].to_numpy(), series["glucose_raw"].to_numpy()
            self.tests.append({
                "id": pid, "group": info["group"], "times": series.index[anchors], "overlap": [o for _, o in meals],
                "own": self.prior_stats[pid] if held_out is None else model.stats(hist, base), "X": X[:, :24],
                "actual": raw[anchors[:, None] + np.arange(1, 25)] - g[anchors][:, None]})    # truth = raw CGM
        model.fit_priors([(None, {"id": p, "group": self.groups[p]}) for p in self.prior_stats], stats=self.prior_stats)

    def score(self, group_strength, person_strength, loose_factor, variants=("flat", "population", "group", "person"),
              with_sd=False):
        """One row per (test meal, variant) with errors at 30/60/90/120 min.
        with_sd: also the raw (uncalibrated) model sd of the person variant at every step."""
        m = self.model.set_strengths(group_strength, person_strength, loose_factor)
        rows = []
        for t in self.tests:
            ex = m.person_stats.get(t["id"], Stats(m.n_features))   # take them out of the priors first
            population = solve(m.total - ex, np.zeros(m.n_features), 100.0, m.scale)
            group = solve(m.group_stats[t["group"]] - ex, population, group_strength, m.scale)
            coef, cov, s2 = posterior(t["own"], group, person_strength, m.scale, m.s2_population)
            coefs = {"flat": np.zeros(m.n_features), "population": population, "group": group, "person": coef}
            sd = np.sqrt(row_variance(t["X"], cov) + s2) if with_sd else None
            for name in variants:
                pred = t["X"] @ coefs[name]
                for k in range(len(pred)):
                    rows.append({"id": t["id"], "group": t["group"], "variant": name, "overlap": t["overlap"][k],
                                 "time": t["times"][k], **{f"err{s * 5}": pred[k, s - 1] - t["actual"][k, s - 1] for s in MEAL_STEPS},
                                 "rise_pred": max(pred[k].max(), 0), "rise_actual": max(t["actual"][k].max(), 0),
                                 "pred": pred[k], "actual": t["actual"][k],
                                 "sd": sd[k] if sd is not None and name == "person" else None})
        return pd.DataFrame(rows)


def top_spikes(res):
    """Each person-day's biggest spike: how far off was the predicted peak rise (bias and abs error)."""
    res = res.assign(day=pd.to_datetime(res["time"]).dt.normalize())
    top = res.loc[res.groupby(["variant", "id", "day"])["rise_actual"].idxmax()]
    err = top["rise_pred"] - top["rise_actual"]
    return pd.DataFrame({"top_spike_bias": err.groupby(top["variant"]).mean(),
                         "top_spike_mae": err.abs().groupby(top["variant"]).mean()})


def summarize(res):
    """RMSE (what we minimize) and MAE at each scored point, per variant, plus the daily top-spike error."""
    by = res.groupby("variant", sort=False)
    out = by[ERRS].agg(lambda e: np.sqrt(np.mean(e ** 2))).add_prefix("rmse_")
    out["rmse_all"] = by.apply(rmse)
    out = out.join(by[ERRS].agg(lambda e: np.mean(np.abs(e))).add_prefix("mae_")).join(top_spikes(res))
    out["meals"] = by.size()
    return out.round(1)


def error_band(res):
    """Noise: typical error (RMSE) at each step after a meal, person variant. Gives the 80% band."""
    res = res[res["variant"] == "person"]
    sigma = np.sqrt(np.mean((np.stack(res["pred"]) - np.stack(res["actual"])) ** 2, axis=0))
    return np.concatenate([sigma, np.full(HORIZON - len(sigma), sigma[-1])])


def calibrate(res):
    """Per-horizon factor so that (error / model sd) has unit spread on these meals. Returns HORIZON values."""
    res = res[res["variant"] == "person"]
    sd = np.stack(res["sd"])
    z = (np.stack(res["pred"]) - np.stack(res["actual"])) / np.where(sd > 0, sd, np.nan)
    kappa = np.sqrt(np.nanmean(z ** 2, axis=0))                   # 24 values, 5..120 min
    return np.concatenate([kappa, np.full(HORIZON - len(kappa), kappa[-1])])


def calibrate_band(res, kappa):
    """The 80 % band: 10th / 90th percentile of (truth - prediction) / sd^alpha per horizon, sd = calibrated.
    alpha in 0..1 is picked by the interval score (width + 10 x how far the truth falls outside) on these meals.
    Returns (alpha, q (2, HORIZON))."""
    res = res[res["variant"] == "person"]
    E = np.stack(res["actual"]) - np.stack(res["pred"])
    S = kappa[:24] * np.stack(res["sd"])
    best = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = S ** alpha
        q = np.nanquantile(E / s, [0.1, 0.9], axis=0)
        lo, hi = q[0] * s, q[1] * s
        score = np.nanmean((hi - lo) + 10 * np.clip(lo - E, 0, None) + 10 * np.clip(E - hi, 0, None))
        if best is None or score < best[0]:
            best = (score, alpha, q)
    q = np.concatenate([best[2], np.repeat(best[2][:, -1:], HORIZON - 24, axis=1)], axis=1)
    return best[1], q


def coverage(res, kappa, alpha, q):
    """How often the truth lands inside the calibrated 80 % band at 30/60/90/120 min, and its average width."""
    res = res[res["variant"] == "person"]
    E = np.stack(res["actual"]) - np.stack(res["pred"])
    s = (kappa[:24] * np.stack(res["sd"])) ** alpha
    lo, hi = q[0][:24] * s, q[1][:24] * s
    inside = (E >= lo) & (E <= hi)
    return {**{f"cov{st * 5}": float(inside[:, st - 1].mean()) for st in MEAL_STEPS},
            "band_width": float(np.mean(hi - lo))}


def tune(split):
    """Score every shrinkage setting on the tuning split; return the best and the full grid."""
    grid = pd.DataFrame([{"group_strength": gs, "person_strength": ps, "loose_factor": lf,
                          "rmse": rmse(split.score(gs, ps, lf, variants=("person",)))} for gs, ps, lf in GRID])
    best = grid.loc[grid["rmse"].idxmin()]
    return (best["group_strength"], best["person_strength"], best["loose_factor"]), grid


def meal_by_meal(model, people, held_out, prior_stats, setting, max_meals=40):
    """Priors without the person. Day 1 sets their baseline. Then predict meal n+1 from meals 1..n."""
    groups = {i["id"]: i["group"] for _, i in people}
    model.fit_priors([(None, {"id": p, "group": groups[p]}) for p in prior_stats], stats=prior_stats)
    model.set_strengths(*setting)
    rows = []
    for series, info in people:
        if held_out is not None and not held_out(info["id"]):
            continue
        day1 = split_days(series, 1)[0]
        base = baseline(day1)
        prior = model.prior(info["group"], exclude=info["id"] if info["id"] in prior_stats else None)
        raw = series["glucose_raw"].to_numpy()
        rows_ = model.rows(series, base)                    # built once; each step only masks what is unseen
        for n, (a, _) in enumerate(scored_meals(series, start=len(day1))[:max_meals]):
            coef = model.personalize(series, info["group"], base, until=a, prior=prior, rows=rows_)
            pred = model.predict_from(series, a, coef, base)[0]
            err = [pred[s - 1] - raw[a + s] for s in MEAL_STEPS]
            rows.append({"id": info["id"], "meals_seen": n, "rmse": float(np.sqrt(np.mean(np.square(err))))})
    return pd.DataFrame(rows)


def plot_meal_by_meal(curves, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for key, df in curves.items():
        m = df.groupby("meals_seen")["rmse"].mean().rolling(5, center=True, min_periods=1).mean()
        ax.plot(m.index, m, color=COLORS[key], lw=2, label=f"{key[0]} {key[1]} ({df['id'].nunique()} people)")
    ax.set_xlabel("Meals already seen for this person", color=INK)
    ax.set_ylabel("RMSE on the next meal, 30-120 min (mg/dL)", color=INK)
    ax.set_title("Learning a never-seen person, meal by meal (5-meal rolling mean)", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=MUTED, alpha=0.15, lw=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def keep_others(path, names, **read):
    """Rows of a saved results file that belong to backends we are not retraining."""
    if not path.exists():
        return []
    df = pd.read_csv(path, **read)
    return [df[~df["backend"].isin(names)]]


def main(only=None):
    RESULTS.mkdir(exist_ok=True)
    WEIGHTS.mkdir(exist_ok=True)
    backends = [b for b in BACKENDS if only is None or b.NAME in only]
    names = [b.NAME for b in backends]
    bundle, tuning, scores, per_meal, curves, shapes_log = {}, [], [], [], {}, {}
    if only is not None:                              # keep the other backends' saved weights and results
        with open(WEIGHTS / "models.pkl", "rb") as f:
            bundle = {k: v for k, v in pickle.load(f).items() if k[0] not in names}
        tuning = keep_others(RESULTS / "tuning.csv", names)
        scores = [s.set_index("variant") for s in keep_others(RESULTS / "scores.csv", names)]
        per_meal = keep_others(RESULTS / "test_errors.csv", names)
        old = keep_others(RESULTS / "meal_by_meal.csv", names)
        if old:
            curves = {k: d.drop(columns=["backend", "kind"]) for k, d in old[0].groupby(["backend", "kind"])}
        if (RESULTS / "overlay_shapes.json").exists():
            shapes_log = {k: v for k, v in json.loads((RESULTS / "overlay_shapes.json").read_text()).items() if k not in names}
    for backend in backends:
        people = backend.load_all()
        held = held_out_rule(backend)
        train_people = [(s, i) for s, i in people if held is None or not held(i["id"])]
        n_test = len(people) - len(train_people) if held else len(people)
        log(f"===== {backend.NAME}: {len(people)} people loaded, {n_test} tested as never-seen =====")

        # Overlay shapes: fit once on training people (metabolic: days 1-7 only, so test days stay unseen)
        shape_data = [(split_days(s, 7 if held is None else SHAPE_DAYS)[0], i) for s, i in train_people][:SHAPE_PEOPLE]
        log(f"fitting overlay shapes on {len(shape_data)} people...")
        shapes = Overlay(backend.OVERLAY_EVENTS).fit_shapes(shape_data, SHAPE_EVALS, log=log, food_evals=FOOD_EVALS)
        shapes_log[backend.NAME] = {e: {k: np.round(np.asarray(list(v.values()) if isinstance(v, dict) else v, float), 2).tolist()
                                        for k, v in d.items()} for e, d in shapes.describe().items()}
        log(f"overlay shapes: {shapes_log[backend.NAME]}")

        for kind in KINDS:
            model = make(kind, backend, shapes)
            log(f"[{kind}] building data summaries...")
            full_stats = {i["id"]: model.stats(s, baseline(s)) for s, i in train_people} if held else None
            tune_split = Split(model, people, held, train_days=5, test_days=2, prior_stats=full_stats)
            setting, grid = tune(tune_split)
            tuning.append(grid.assign(backend=backend.NAME, kind=kind))
            tune_res = tune_split.score(*setting, variants=("person",), with_sd=True)
            kappa = calibrate(tune_res)
            alpha, band_q = calibrate_band(tune_res, kappa)
            log(f"[{kind}] tuned (group, person, loose) = {setting}, tuning RMSE {grid['rmse'].min():.2f}; "
                f"sd calibration x{kappa[5]:.1f} (30 min) .. x{kappa[23]:.1f} (120 min); band: sd^{alpha}, "
                f"120 min from {band_q[0][23]:.1f} to +{band_q[1][23]:.1f}")
            res = Split(model, people, held, train_days=7, prior_stats=full_stats).score(*setting, with_sd=True)
            cov80 = coverage(res, kappa, alpha, band_q)
            table = summarize(res)
            for c, v in cov80.items():
                table.loc["person", c] = round(v * 100, 1) if c.startswith("cov") else round(v, 1)
            log(f"[{kind}] test-day coverage of the 80% band: " + ", ".join(f"{c[3:]} min {v:.0%}" for c, v in cov80.items()
                                                                          if c.startswith("cov")) + f"; average width {cov80['band_width']:.1f} mg/dL")
            log(f"[{kind}] test, learn days 1-7 -> predict day 8+ (mg/dL):\n"
                + table.drop(columns=[c for c in table if c.startswith("mae_")]).to_string())
            scores.append(table.assign(backend=backend.NAME, kind=kind))
            per_meal.append(res[res["variant"] == "person"].drop(columns=["pred", "actual", "sd"]).assign(backend=backend.NAME, kind=kind))
            res.to_pickle(RESULTS / f"test_curves_{backend.NAME}_{kind}.pkl")

            prior_stats = full_stats or {i["id"]: model.stats(s, baseline(s)) for s, i in people}
            curves[(backend.NAME, kind)] = meal_by_meal(model, people, held, prior_stats, setting)
            m = curves[(backend.NAME, kind)].groupby("meals_seen")["rmse"].mean()
            log(f"[{kind}] meal by meal RMSE: 0 meals {m.iloc[0]:.1f} | 1-5: {m.iloc[1:6].mean():.1f} | "
                f"6-15: {m.iloc[6:16].mean():.1f} | 16-40: {m.iloc[16:].mean():.1f}")

            final = make(kind, backend, shapes)                      # final: priors from everyone, all days
            everyone = dict(prior_stats)
            everyone.update({i["id"]: final.stats(s, baseline(s)) for s, i in people if i["id"] not in everyone})
            final.fit_priors(people, stats=everyone).set_strengths(*setting)
            final.sigma, final.setting, final.kappa = error_band(res), setting, kappa
            final.band_alpha, final.band_q = alpha, band_q
            final.tune_rmse = float(grid["rmse"].min())          # the site uses each backend's better model by this
            if held is not None:          # keep per-person summaries only for people who can be shown as never-seen
                final.person_stats = {p: st for p, st in final.person_stats.items() if held(p)}
            bundle[(backend.NAME, kind)] = final
            log(f"[{kind}] final model fit on all {len(people)} people")

    with open(WEIGHTS / "models.pkl", "wb") as f:
        pickle.dump(bundle, f)
    pd.concat(tuning).to_csv(RESULTS / "tuning.csv", index=False)
    pd.concat(scores).to_csv(RESULTS / "scores.csv")
    pd.concat(per_meal).round(2).to_csv(RESULTS / "test_errors.csv", index=False)
    pd.concat([df.assign(backend=k[0], kind=k[1]) for k, df in curves.items()]).to_csv(RESULTS / "meal_by_meal.csv", index=False)
    (RESULTS / "overlay_shapes.json").write_text(json.dumps(shapes_log, indent=2))
    plot_meal_by_meal(curves, RESULTS / "meal_by_meal.png")
    log("Saved weights to models/weights/, results to results/")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
