# Plan

## 1. The big picture

**Goal:** from a person's past days (glucose, meals, activity, insulin), predict their glucose after a new meal.
Scored on the error at 30 / 60 / 90 / 120 min after each meal (least squares, so RMSE).

```
patient history (any number of days)               a new meal (typed, CSV, or food photo)
        |                                                    |
        v                                                    v
 classifier.py: type + backend                  photo2nutrition.py: carbs, protein, fat, fiber, kcal
        |                                                    |
        +-------------------------+--------------------------+
                                  v
   clean.py (remove sensor jitter)  ->  context.py (meal context) + events.py (exercise, quiet time)
                                  v
   backend: metabolic.py (healthy / prediabetes / t2d, CGMacros)   or   t1d.py (type 1, MetaboNet parquet)
                                  v
             exogenous.py: inputs -> effect curves (OpenAPS insulin curve)
                                  v
      Model 1: regressor.py                       Model 2: overlay.py
      time series regression (ARX)                baseline + 3 spike and 3 cooldown sigmoids per event
                                  v
      common.py: least squares, population -> group -> person, uncertainty (posterior covariance)
                                  v
      prediction for any window + calibrated 80 % band;  decompose.py: baseline + residue + pieces
                                  v
                 website (app.py, views/), showcase/ demo patients, results/
```

## 2. Data

| | Metabolic backend | Type 1 backend |
|---|---|---|
| Who | healthy, prediabetes, type 2 | type 1 (makes no insulin) |
| Source | CGMacros, 45 people, ~10 days | MetaboNet parquet: Loop, PEDAP, AZT1D, HUPA-UCM, OhioT1DM |
| People used | all 45 | 382 with >= 60 % CGM, >= 14 meals, >= 14 boluses in their first 28 days |
| Train / test | every person tested, each removed from the priors first | 334 train, 48 never-seen test (all OhioT1DM + 10 % of the rest) |
| Medication | none (own insulin estimated from carbs) | bolus + basal doses |
| Meals | carbs, protein, fat, fiber, kcal, photos | carbs |

Why not all 1,414 parquet people: ~540 are in studies with no carb logs, ~480 Loop users log almost no meals in
their first 28 days. Adding them would teach the model that insulin raises glucose (it is given at meals the
model cannot see).

**Cleaning** (`clean.py`): a wiggle smaller than 1 % of the level is treated as sensor jitter and flattened.
Removes ~26 % of turning points in raw type 1 CGM, ~12 % in CGMacros (Libre is already smooth).
Models train on cleaned glucose; every score is against the raw readings.

## 3. Routing (`classifier.py`)

Stated type > takes mealtime insulin (type 1) > lab A1c > logistic regression on 7 days of CGM trends.
Leave-one-out accuracy from glucose alone: 67 % (simple "A1c from mean glucose" rule: 49 %), all 12 type 1 found.
Mistakes are only between neighbouring groups, which share the metabolic backend.

## 4. What the models know

- **Meal** (`context.py`): carbs, protein, fat, fiber, kcal; meal number of the day; clock time; carbs and kcal
  eaten earlier that day; time since the last meal; exercise in the 2 h before; glucose vs usual; the person's
  usual level and swings (a stand-in for medical history); their own past spike per 10 g carbs for that meal slot;
  very large meals. All from before the meal.
- **Non-meal events** (`events.py`): exercise sessions (activity z > 2 for 15+ min), split into after a meal and on
  its own; quiet 4-hour blocks with no food or exercise (night, dawn, morning, afternoon, evening, late).
  Evidence: in type 1, exercise on its own drops glucose ~26 mg/dL in an hour; CGMacros nights drift down ~11 mg/dL.
- **Insulin** (type 1): doses through the open-source OpenAPS insulin action curve.

## 5. The two models

**Regressor** (`regressor.py`): from the last known glucose, future change = recent trend + pull back to the
usual level + each input convolved with fixed response shapes x learned weights + time of day. Linear, so
ridge least squares. Later versions, each tested on never-seen people before adopting (regressor RMSE,
metabolic / type 1 with 100 training people):
- v5, more from the glucose history: acceleration (last 15 min vs the 15 before), the last hour's trend, distance
  above 180 / below 70, the person's own daily rhythm (median glucose per clock time over earlier days).
  23.3 -> 22.8 / 42.0 -> 41.6.
- Wider shrinkage grid (the best settings had been on the grid's edge): 22.8 -> 22.5.
- v6, what else is in the meal: a meal is any food (not just carbs), carbs x fat share, x protein share, x fiber,
  and a late (4 h) bump for fat and protein. Even on test, better on the tuning days. (Type 1 data logs carbs only.)
- v7, the person's peaks and lows: swings over their whole history, usual daily peak and low vs usual level.
  22.5 -> 22.4 / tuning days 46.1 -> 45.9.
- Tried: modelling glucose minus the person's usual daily curve (weight fixed at 1) instead of learning the curve's
  weight: identical (22.40 both); the learned net weight is about 0.14.

**Overlay** (`overlay.py`): baseline (trend, pull, rhythm, exercise, HR) + for every event 3 spike sigmoids and
3 cooldown sigmoids. Each event type (meal, insulin, exercise, quiet) has its own sigmoid centers and widths.
Sizes are a linear mix of the event's inputs and context. Each meal also gets its own speed (time-warp) from its
size, fat / protein / fiber share, clock time, time since the last meal and exercise. Shapes are fit with scipy
Powell, one event type at a time; sizes by least squares.

Both learn in three layers (population -> group -> person); shrinkage strengths are tuned on days 1-5 -> 6-7
over 100 settings. The site uses, per backend, the model with the lower tuning-day error (`tune_rmse`), never test
results. How strongly each parameter goes with the rise after a meal is measured with distance correlation (dcor)
in `explain_data.py`, shown on How it works.

## 6. Uncertainty

- Person fit: coefficients ~ covariance = s2 x (X'WX + P)^-1, s2 = residual variance (mean squared residual per point).
  With little own data, s2 is blended toward the population's.
- Any curve that is linear in the coefficients (every spike, cooldown, sum, the residue) gets variance x' cov x.
  Prediction adds s2. Long windows carry each new starting point's variance forward.
- Raw bands are too narrow (neighbouring 5-min points are correlated), so they are scaled per horizon until
  error / sd has unit spread on days 6-7 (chi-squared per point = 1).
- The 80 % band: errors after meals are skewed (big spikes run higher than predicted) and heavy-tailed, so the
  band is the 10th / 90th percentile of (truth - prediction) / sd^alpha per horizon, alpha in 0..1 picked by the
  interval score, all on the tuning days. Before (mean +- 1.28 sd) it held the truth 84-87 % of the time, i.e. was
  too wide; on a split of the test people the new band holds ~80-82 % and is 11-12 % narrower.
  Tried and dropped: scaling each person's band by their own past errors, widening with the size of the predicted
  change (no gain, or worse for type 1).
- **Personal band (site, first 3 h after a meal):** replay the person's last 20 meals one by one, as if live
  (learn from everything before the meal, predict it, note the miss), pool those misses with 10 population misses
  (`band_pop.py`), band = 10th / 90th percentile per step. Tested on never-seen metabolic people (days 2-7 replayed,
  day 8+ scored): 80 % band holds 80.1 % at width 48.0 vs 82.4 % at 51.7 for the population band; 90 % band
  88.8 % at 62.7 vs 90.5 % at 69.5. Own misses only: narrower but 78 % / 86 % (too few meals in a week).
  The same replay gives the "off by" number shown with every prediction. Full days keep the calibrated model band.
  Type 1 (48 never-seen people): the personal band held only 72-74 % (85 mg/dL wide) vs 82 % for the population
  band (96 wide); with 30 population misses it was no narrower. Earlier misses don't foretell later ones in type 1,
  so type 1 keeps the calibrated model band.
- Not included: group-level coefficient uncertainty and the overlay shape parameters (treated as fixed).

## 7. Results (never-seen patients, learn days 1-7, predict day 8+, RMSE mg/dL vs raw CGM)

| | 30 | 60 | 90 | 120 | all | biggest-spike bias | 80 % band coverage (30/60/90/120) |
|---|---:|---:|---:|---:|---:|---:|---|
| Metabolic, flat line (no model) | 27.9 | 41.3 | 40.5 | 38.0 | 37.3 | -65.9 | |
| Metabolic regressor, person | 19.1 | 25.1 | 24.4 | 24.1 | **23.3** | -28.3 | 81 / 83 / 84 / 86 % |
| Metabolic overlay, person | 19.1 | 26.1 | 25.1 | 24.9 | 24.0 | -29.7 | 81 / 83 / 84 / 83 % |
| Type 1, flat line | 30.4 | 50.1 | 58.9 | 64.4 | 52.6 | -88.5 | |
| Type 1 regressor, person | 27.2 | 41.1 | 46.3 | 49.6 | **41.9** | -50.6 | 87 / 87 / 88 / 87 % |
| Type 1 overlay, person | 28.4 | 40.8 | 46.0 | 49.5 | **41.9** | -47.3 | 87 / 88 / 88 / 87 % |

553 metabolic and 3,845 type 1 test meals. Full tables (population / group / person): `results/scores.csv`.

**Learning a new person meal by meal** (RMSE on the next meal):

| meals seen | 0 | 1-5 | 6-15 | 16-40 |
|---|---:|---:|---:|---:|
| Metabolic regressor | 36.9 | 23.3 | 22.5 | 18.7 |
| Metabolic overlay | 39.4 | 23.2 | 22.7 | 18.7 |
| Type 1 regressor | 41.2 | 39.2 | 40.0 | 37.8 |
| Type 1 overlay | 41.2 | 39.3 | 39.1 | 37.7 |

**Versions** (each tuned on its own; type 1 with 100 training people for speed):

| RMSE | v1 | v2 | v3 | v4 |
|---|---:|---:|---:|---:|
| Metabolic regressor | 23.5 | 24.0 | 23.8 | **23.3** |
| Metabolic overlay | 25.3 | 24.4 | 25.4 | 24.8 |
| Type 1 regressor | 42.3 | 42.4 | 42.2 | **42.0** |
| Type 1 overlay | 42.2 | 42.1 | 41.8 | **41.7** |

v1 meals only; v2 + cleaning, meal number, day so far, exercise, pre-meal glucose, own past spikes; v3 + kcal,
clock time, gap since last meal, usual level, swings, large meals, overlay time-warp; v4 + exercise and quiet-time
events. Biggest-spike bias improved every version (metabolic overlay -38.6 -> -25.4; type 1 ~-54 -> -49).
More training people helps type 1: Ohio test patients 47.0 (priors from 11 Ohio people) vs 45.4 (100 people).

**Photo to nutrition** (`photo2nutrition.py`, all open): CLIP ViT-L/14 recognises the photo zero-shot against
the ~5,200 foods of USDA FNDDS 2021-2023 (public domain); nutrition = confidence-weighted top matches at their
typical portion ("quantity not specified" weight from the US national diet survey); blended with a photo memory of
logged meals (closer match counts more). Memory curve and lookup scale tuned on days 1-5 -> 6-7, test on day 8+:
see `results/photo2nutrition_scores.csv`. The site adds a 0-3x portion slider, shows the 3 closest logged meal
photos (cosine similarity of CLIP image vectors, no labels) and lets you use one's logged numbers.
Speed: only CLIP's image half is used (text vectors cached), from a local half-precision copy (1 s to load instead
of 13), and through OpenVINO on the Intel GPU when there is one (0.13-0.2 s a photo vs 1.5 s with PyTorch, no
PyTorch import; vectors match PyTorch's at >= 0.9995). Demo photos' analyses are saved with the demo files.

**Tried and dropped:** gradient boosted trees on the 4 points (no continuous curve), predicting each meal's
milestones directly (errors compound), trees correcting the regressor (helped one backend, hurt the other).

## 8. Website: Glucklich (`streamlit run app.py`)

*Glucklich*, from the German for *happy*. Top navigation, five pages:

- **Home:** the name (with a button that says it), Gray's Anatomy plates of the pancreas and its islets (1918,
  public domain, paper removed: light ink in dark mode, the same with lightness flipped in light mode), one real
  day of glucose as line art, three facts, and a three-step demo with no settings: Maya (prediabetic, her real meal
  photo), Ana (type 1) or Sam (type 2, his real photo), Predict!, then "See what really happened" with the average error, RMSE and band coverage.
  Demo people are learned once for every visitor, so switching between them is instant.
- **Meal:** drop files (history, about me, meal, photo, insulin plan, exercise) or drag a sample person onto the
  box; pick a new meal or one already in the file; read the plate (CLIP's top 3 dishes or search all 5,200 FNDDS
  foods, several foods per plate, a portion each, blended with the photo memory) or type it; any exercise or insulin
  in the next 3 h; Predict!; then Validate! against the file's own later readings or a dropped file.
- **Full Day:** same step 1; plan the day (meals with their insulin, exercise, other insulin shots); Predict! draws
  the whole day with key insights (peak and its meal, biggest spike, time in range, lows and why, exercise effect,
  time above 180); step 3 validates; download the prediction, go to Meal, or How it works. Any day in the file can
  be picked: the plan is filled from that day's meals, insulin and exercise and its glucose is kept for Validate!.
- **Log:** start from your own files (CSV or Excel, any shape parse.py reads) and/or type or paste glucose, meals,
  exercise, insulin and a profile in editable tables; download the CSVs or send them to Meal / Full Day.
- Commentary under every meal prediction (`engine.meal_commentary`): peak time and size with the band's range, how it
  compares with the person's own past rises after similar carbs, when it is back near the start, dips, time above
  180, possible lows. Full days: `day_insights`.
- Demo and sample people (`choose_demos.py`): every never-seen person's meals predicted with the site's model, people
  ranked by their median meal error; Home: Maya (CGMacros 010, prediabetic), Sam (CGMacros 028, type 2), Ana (Loop 116,
  type 1); samples: Jo (CGMacros 017, healthy) and Theo (Loop 1163, type 1). Their photos are the CGMacros image
  logged on that exact meal row. Demo meals (v7, personal band): Maya RMSE 7, Sam 5, Ana 10; truth 100 % in band.
- Personal band per backend (`engine.BAND_METHOD`): metabolic "own misses + 10 borrowed" (the tightest that held
  >= 80 % on later meals: 80.1 % at 48 mg/dL); type 1 the calibrated model band (no personal version was tighter
  while holding 80 %: scaled 81.6 % at 97 vs 81.9 % at 96).
- Charts: glucose axis 60-200 mg/dL, widened to the next 5 only if the data or band go beyond; ticks every 20.
- Files: `parse.py` reads our format, raw CGMacros files, Dexcom / Libre / Nightscout exports, separate date and
  time columns, unix times, mmol/L. The model only learns from data before the predicted moment. After Validate!, the
  meal or day joins the person's history for their next prediction and is saved to `data/contributed/`.
- **How it works:** the question (Sam's lunch), then parts revealed one by one: getting to know Sam (sensor photos,
  Wikimedia Commons), cleaning and classification, what's in the meal, food and insulin in the blood, the model
  (every parameter with Sam's values, distance correlations, the regressor's pushes, the overlay's spikes and
  cooldowns, which model is used and why, population -> group -> Sam), how sure we are (the personal band),
  validation and learning from it, results by type of person, the building blocks, references. Colour-coded with
  one site-wide table (`style.ROLES`: glucose red, food yellow, insulin orange, activity purple, prediction magenta,
  truth green, model blue, open tools cyan; each with a light and a dark Flexoki value).

Speed: nothing heavy runs on page load. Home draws its first screen from `style.py` alone (plates inline as small
WebP masks coloured by CSS, so one copy serves both themes); a person is learned only when Predict! is pressed and
kept for every visitor; the 1.7 GB photo model loads only when a photo is read; How it works learns Sam only at the
chapter that needs it. Streamlit's Deploy button is hidden (`toolbarMode = "viewer"`).

Colours are Flexoki for whichever theme the page shows (light: 600 accents, base-600/300 greys; dark: 400 accents,
base-500/700 greys). Switching theme from Streamlit's menu does not rerun the page, so a small watcher reads the
page background, swaps the colours and plates at once, and reruns once so the charts follow.

Exercise is added as activity on top of the person's usual pattern (`plan.py`), so the meal and day pages and the
demo pickers share one setup. Sample sets: `try_me/<person>/{meal, meal_check, day, day_check}`, copied with zips
to `static/try_me/` so the cards can be dragged and downloaded.

| | Person | Data | Picked | RMSE vs truth |
|---|---|---|---|---:|
| Home demo | Maya, prediabetic | CGMacros 041 | best meal (`pick_demos.py`) | 7 |
| Home demo | Ana, type 1 | OhioT1DM 588 | best meal | 7 |
| Home demo + sample | Sam, type 2 | CGMacros 014 | best meal / best day (`pick_days.py`) | 4.7 / 20 |
| Sample | Theo, type 1 | PEDAP 61 | best meal / best day | 9 / 30 |

All chosen on purpose (the page says so) and taken out of the population and group priors, so they stay never-seen.

## 9. Files

```
app.py                      router (top navigation, four pages)
views/home.py               landing page and three-step demo
views/meal.py               one meal: drag and drop, meal + exercise + insulin, predict, validate
views/full_day.py           a whole day: plan, predict, insights, validate, download
views/engine.py             shared: loading, sample cards, file sorting, learning, predicting, scoring, plots, insights
views/style.py              Flexoki colours, shared CSS, theme watcher, the Glucklich name (light: draws the first screen)
views/how_it_works.py       plain-language explanation
archive/                    the older all-in-one tool, its timeline and the old showcase (not used by the site)
data/downloads/             the original dataset downloads (zips and extracted copies)
models/
  common.py                 data format, least squares, 3-layer shrinkage, posterior + uncertainty, window prediction
  clean.py                  jitter removal
  context.py                meal context
  events.py                 exercise and quiet-time events
  exogenous.py              inputs -> effect curves (OpenAPS insulin curve)
  plan.py                   a plan (meals, exercise, insulin) -> model inputs for the hours ahead
  metabolic.py, t1d.py      the two backends (loaders, inputs)
  classifier.py             type + backend routing
  regressor.py, overlay.py  the two models
  decompose.py              prediction = baseline + residue = ... with uncertainty
  photo2nutrition.py        food photo -> nutrition
  train.py                  tune, calibrate, test, meal-by-meal, save weights (python models/train.py [backend])
  compare.py                version comparison
  pick_demos.py             finds the best predicted never-seen meals for the demos
  pick_days.py              finds the best predicted whole days for the sample people
  explain_data.py           numbers for How it works (average meal response per type, Sam's photo)
  demo_files.py             builds samples/home_demo and samples/try_me
  other_candidates.py       builds samples/test_people
  choose_demos.py           picks the demo and sample people
  band_pop.py               population misses for the personal band
  weights/                  models.pkl, classifier.pkl
data/                       datasets, USDA FNDDS (data/fndds, fndds_foods.csv), caches
results/                    scores, versions, tuning, meal-by-meal, logs
static/fonts/               Libron (OFL)
static/art/                 Gray's Anatomy plates 1100 and 1105 (public domain), cutout.py, glucose_day.csv
static/try_me/              sample sets served to the site (built by demo_files.py)
samples/home_demo/           the Home demo people
samples/try_me/              drag-and-drop sample sets (copied with zips to static/try_me/)
samples/test_people/         10 random people (never seen and trained on) to test drag and drop (other_candidates.py)
```

## 10. How to run

```
conda activate openpayments
python models/train.py            # both backends (~35 min); or: python models/train.py t1d
python models/classifier.py
python models/photo2nutrition.py
python models/explain_data.py
python models/compare.py          # optional, ~45 min
streamlit run app.py
```

## 11. Limitations and next steps

- Real data; the judging data is synthetic.
- Type 1 is hard (RMSE ~42): self-reported carbs, unlogged snacks, closed-loop insulin that reacts to glucose.
- Small metabolic dataset (45 people, ~7 days each to learn from).
- The day's biggest spike is still under-predicted (~25-30 mg/dL metabolic), partly because picking the largest
  spike also picks the ones that ran high by chance.
- Next: use each Loop user's best-logged 28 days (more type 1 people); per-person band width; group-level
  uncertainty in the bands.
