"""Files for the home page demo (demo/) and ready-made sets to drag and drop on the site (try_me/).

The people, meals and days are the best predicted ones from pick_demos.py and pick_days.py (chosen on purpose).
  samples/home_demo/<name>/                 about_me, history (up to the meal), meal (+ photo), doses (type 1), truth
  samples/try_me/<name (type)>/meal/        1_about_me, 2_my_glucose_history, 3_my_meal (+ photo, + insulin plan for type 1)
  samples/try_me/<name (type)>/meal_check/  4_what_really_happened (glucose in the 3 h after the meal)
  samples/try_me/<name (type)>/day/         1_about_me, 2_my_glucose_history (up to midnight), 3_my_day_meals,
                                    3_my_day_exercise, 3_my_day_insulin (type 1)
  samples/try_me/<name (type)>/day_check/   4_what_really_happened_that_day
static/art/glucose_day.csv is the real day drawn on the home page.
(all under samples/) try_me/ is also copied to static/try_me/ (with a zip per set) so the site can drag and download it.

Run after train.py, pick_demos.py and pick_days.py:  python models/demo_files.py
"""
import shutil

import pandas as pd
from PIL import Image

import metabolic
import t1d
from common import ROOT
from pick_days import day_plan

SAMPLES = ROOT / "samples"

H3 = pd.Timedelta(hours=3)
DAY = pd.Timedelta(days=1)
TYPE_NAME = {"healthy": "healthy", "prediabetes": "prediabetic", "t2d": "type 2 diabetes", "t1d": "type 1 diabetes"}
# name, folder, backend, dataset id, meal time, whole day (try_me only)
def people():
    """From choose_demos.py (results/demo_people.json) and pick_days.py: (name, folder, backend, id, meal time, day)."""
    import json
    chosen = json.loads((ROOT / "results" / "demo_people.json").read_text())
    days = pd.read_csv(ROOT / "results" / "day_candidates.csv")
    out = []
    for name, (where, backend, pid) in chosen.items():
        day = days[days["id"] == pid].sort_values("rmse").iloc[0]["day"] if where == "try_me" else None
        out.append((name, where, backend, int(pid.split("-")[1]) if backend == "metabolic" else pid, None, day))
    return out


def meal_name(t):
    h = t.hour + t.minute / 60
    return "Snack" if h < 4 or h >= 21.5 else "Breakfast" if h < 11 else "Lunch" if h < 16 else "Dinner"


def photo_for(subject, t0):
    """The CGMacros photo logged within the meal's 5-minute slot, if any."""
    raw = pd.read_csv(metabolic.CGMACROS / f"CGMacros-{subject:03d}" / f"CGMacros-{subject:03d}.csv", parse_dates=["Timestamp"])
    raw.columns = raw.columns.str.strip()
    hit = raw[raw["Meal Type"].notna() & (raw["Timestamp"] >= t0) & (raw["Timestamp"] < t0 + pd.Timedelta(minutes=5))]
    path = hit["Image path"].dropna()
    return metabolic.CGMACROS / f"CGMacros-{subject:03d}" / path.iloc[0] if len(path) else None


def history_csv(raw, until, path):
    h = raw[raw.index < until]
    h = h[[c for c in h.columns if c == "glucose" or h[c].fillna(0).abs().sum() > 0]]
    h.rename_axis("time").reset_index().to_csv(path, index=False)


def glucose_csv(raw, start, end, path):
    raw[["glucose"]][(raw.index >= start) & (raw.index < end)].rename_axis("time").reset_index().to_csv(path, index=False)


def fresh(folder):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    return folder


def build(name, where, backend, pid, when, day, out=None):
    """out: the folder for try_me-style sets (default try_me_building)."""
    series, info = metabolic.load(pid) if backend == "metabolic" else t1d.load(pid)
    if when is None:                                  # the meal pick_demos.py ranked best for this person
        cands = pd.read_csv(ROOT / "results" / "demo_candidates.csv", parse_dates=["time"])
        c = cands[(cands["id"] == info["id"]) & (cands["true_rise"] >= 30) & (cands["carbs"] >= 20)]
        if backend == "metabolic":                    # with a photo for the demo
            c = c[[photo_for(pid, t) is not None for t in c["time"]]]
        when = c.sort_values(["rmse", "rise_error"]).iloc[0]["time"]
    t0 = pd.Timestamp(when)
    raw = series.assign(glucose=series["glucose_raw"]).drop(columns="glucose_raw")     # files hold the raw sensor
    r = raw.loc[t0]
    meal = pd.DataFrame([{"time": t0.strftime("%Y-%m-%d %H:%M"), "meal": meal_name(t0),
                          **{c: round(float(r[c]), 1) for c in ["carbs", "protein", "fat", "fiber", "calories"]},
                          "insulin": round(float(r["bolus"]), 2)}])
    later = raw["bolus"][(raw.index > t0) & (raw.index <= t0 + H3)]
    doses = pd.DataFrame({"time": later[later > 0].index.strftime("%Y-%m-%d %H:%M"), "units": later[later > 0].round(2).to_numpy()})
    about = pd.DataFrame([{"name": name, "diabetes_type": info["group"], "a1c": info.get("a1c", ""), "age": info.get("age", ""),
                           "source": info["source"], "dataset_id": info["id"]}])
    photo = photo_for(pid, t0) if backend == "metabolic" else None

    def save_photo(path):
        if photo is not None and photo.exists():
            img = Image.open(photo)
            img.thumbnail((900, 900))
            img.save(path, quality=88)

    if where == "demo":
        f = fresh(SAMPLES / "home_demo_building" / name)
        about.to_csv(f / "about_me.csv", index=False)
        history_csv(raw, t0, f / "history.csv")
        meal.to_csv(f / "meal.csv", index=False)
        if backend == "t1d":
            doses.to_csv(f / "doses.csv", index=False)
        glucose_csv(raw, t0, t0 + H3 + pd.Timedelta(minutes=5), f / "truth.csv")
        save_photo(f / "meal_photo.jpg")
        if (f / "meal_photo.jpg").exists():                 # what the photo model sees, so the home page needn't load it
            import json
            import photo2nutrition as pn
            res = pn.analyse(str(f / "meal_photo.jpg"))
            res["nutrition"] = pn.plate([(res["candidates"][0][0], 1.0)], res["memory"], res["memory_weight"])
            res["names"] = [(pn.food(i)["name"], s_) for i, s_ in res["candidates"]]
            res["options"] = [{"name": pn.food(i)["name"], "nutrition": pn.plate([(i, 1.0)], res["memory"], res["memory_weight"])}
                              for i, _ in res["candidates"]]
            res["candidates"] = [[int(i), float(s_)] for i, s_ in res["candidates"]]
            (f / "photo_analysis.json").write_text(json.dumps(res, indent=1))
    else:
        top = (out or SAMPLES / "try_me_building") / f"{name} ({TYPE_NAME[info['group']]})"
        f = fresh(top / "meal")
        about.to_csv(f / "1_about_me.csv", index=False)
        history_csv(raw, t0, f / "2_my_glucose_history.csv")
        meal.to_csv(f / "3_my_meal.csv", index=False)
        if backend == "t1d":
            doses.to_csv(f / "3_my_insulin_plan.csv", index=False)
        save_photo(f / "3_my_meal_photo.jpg")
        glucose_csv(raw, t0, t0 + H3 + pd.Timedelta(minutes=5), fresh(top / "meal_check") / "4_what_really_happened.csv")

        d0 = pd.Timestamp(day)
        meals, day_doses, sessions = day_plan(series, d0)
        f = fresh(top / "day")
        about.to_csv(f / "1_about_me.csv", index=False)
        history_csv(raw, d0, f / "2_my_glucose_history.csv")
        pd.DataFrame([{"time": m["time"].strftime("%Y-%m-%d %H:%M"), "meal": meal_name(m["time"]),
                       **{c: round(m[c], 1) for c in ["carbs", "protein", "fat", "fiber", "calories"]},
                       **({"insulin": round(m["insulin"], 2)} if backend == "t1d" else {})} for m in meals]
                     ).to_csv(f / "3_my_day_meals.csv", index=False)
        pd.DataFrame([{"start": t.strftime("%Y-%m-%d %H:%M"), "minutes": mins, "intensity": inten} for t, mins, inten in sessions],
                     columns=["start", "minutes", "intensity"]).to_csv(f / "3_my_day_exercise.csv", index=False)
        if backend == "t1d":
            pd.DataFrame({"time": [t.strftime("%Y-%m-%d %H:%M") for t, _ in day_doses], "units": [round(u, 2) for _, u in day_doses]}
                         ).to_csv(f / "3_my_day_insulin.csv", index=False)
        glucose_csv(raw, d0, d0 + DAY, fresh(top / "day_check") / "4_what_really_happened_that_day.csv")
    print(f"{name}: {info['id']} meal {t0} ({r['carbs']:.0f} g carbs, {len(doses)} later doses, "
          f"photo {'yes' if photo is not None else 'no'}){f', day {day}' if day else ''}")


def art_day(subject=11, day="2020-10-05"):
    """The one real day drawn as line art on the home page (glucose and meal times only)."""
    series, _ = metabolic.load(subject)
    d0 = pd.Timestamp(day)
    d = series[(series.index >= d0) & (series.index < d0 + DAY)]
    pd.DataFrame({"time": d.index, "glucose": d["glucose_raw"].round(1), "carbs": d["carbs"]}).to_csv(
        ROOT / "static" / "art" / "glucose_day.csv", index=False)


def publish():
    """Copy try_me/ to static/try_me/ with one zip per set, for dragging and downloading on the site."""
    out = ROOT / "static" / "try_me"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SAMPLES / "try_me", out)
    for person in out.iterdir():
        for part in [p for p in person.iterdir() if p.is_dir()]:
            shutil.make_archive(str(person / f"{person.name} {part.name}"), "zip", part)


if __name__ == "__main__":
    # build next to the live folders and copy them in at the end, so the site never sees them missing
    for folder in ("home_demo_building", "try_me_building"):
        if (SAMPLES / folder).exists():
            shutil.rmtree(SAMPLES / folder)
    for row in people():
        build(*row)
    for folder in ("home_demo", "try_me"):          # copy over, then tidy (Windows can lock a folder the site is reading)
        shutil.copytree(SAMPLES / f"{folder}_building", SAMPLES / folder, dirs_exist_ok=True)
        built = {p.relative_to(SAMPLES / f"{folder}_building") for p in (SAMPLES / f"{folder}_building").rglob("*")}
        for p in sorted((SAMPLES / folder).rglob("*"), reverse=True):    # remove what the new build doesn't have
            if p.relative_to(SAMPLES / folder) not in built:
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        shutil.rmtree(SAMPLES / f"{folder}_building", ignore_errors=True)
    publish()
    art_day()
