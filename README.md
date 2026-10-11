# Glucklich

*Glücklich*, German for *happy*. Built by **Wellness Team #3** for Health Hack 2026.

Show it what's on your plate and it sketches your glucose for the next three hours, or for a whole day: the likely
curve, a band where the real readings will most likely fall, and a few plain notes on what to expect. It learns each
person from their own history first, then fills the gaps from people like them.

![A prediction for Maya's lunch, with what really happened](docs/img/predict_light.webp)

## What it does

- **Meal:** drop your glucose history (our CSV, a CGMacros file, or a Dexcom / Libre / Nightscout export, CSV, Excel
  or JSON) and a meal: typed, from a file, or from a photo. Get the next 3 hours with an 80 % band.
- **Full Day:** plan a day of meals, exercise and insulin and see the whole day, with its peaks, lows and time in range.
- **Tell us how we did:** drop what really happened (one file, many, a whole folder or a zip). The site finds the
  readings for the predicted window, scores the prediction (RMSE and how many readings fell inside the band) and
  learns from it for the next one.
- **How it works:** the whole pipeline, step by step, on one real person. Start here.

## How it works

The site's **How it works** page walks through all of this on one real person (Sam, type 2), with his own data and
charts. In short:

1. **Getting to know you.** A glucose sensor reads every 5 minutes. Add meals, activity and, for type 1, insulin.
2. **Cleaning and classification.** Sensor wiggles under 1 % of the level are smoothed out. A scikit-learn logistic
   regression tells type 1 from everyone else using glucose alone, which picks the model to use.
3. **What's in the meal?** CLIP (an open image model) names the food in a photo. Its 3 best guesses are looked up in
   USDA FNDDS at a typical portion, and blended with meals you logged before that look the same.
4. **Food and insulin in the blood.** Each meal becomes absorption curves (slower with fat, protein and fiber).
   Insulin follows the OpenAPS action curve: real doses for type 1, carbs as a stand-in for everyone else.
5. **The model.** A linear regression: each input's curve times its weight, added up, gives glucose 5 to 240 min
   ahead. The weights start from everyone's, then people like you, and move toward yours only as far as your data
   supports. A second model (S-curves for each rise and fall) is kept for comparison; the site uses whichever did
   better on held-out days.
6. **How sure are we?** Your last 20 meals are predicted again from only what came before each. Their misses, plus
   10 from other people, set the band where 8 in 10 readings should fall. Type 1 uses one calibrated band for everyone.
7. **Validation.** Test people are never seen in training: learn days 1 to 7, predict every meal after, compare with
   the raw sensor. The site follows the same rule, and a validated meal joins your history for the next prediction.

## Results (people the model never saw)

| | RMSE (mg/dL) | Truth inside the 80 % band |
|---|---:|---:|
| Healthy, prediabetes, type 2 | 22.4 | about 80 % |
| Type 1 | 41.6 | about 82 % |

Learn from days 1 to 7, predict every meal from day 8 on, scored against the raw sensor. Details in `Plan.md`.

## Run it

```
git lfs install
git clone https://github.com/AlmostAPhysicist/HealthHack2026
cd HealthHack2026
pip install -r requirements.txt
streamlit run app.py
```

The trained weights ship with the code, so the site runs straight away. The first photo you read downloads CLIP
(about 1.7 GB, once). Retraining needs the datasets themselves (see `AboutDatasets.md`), which are not in this
repository.

## Code

```
app.py                  the site: top navigation over the pages in views/
views/
  home.py               landing page and the three-person demo
  meal.py, full_day.py  predict a meal / a whole day, then validate
  log.py                make your files in editable tables
  how_it_works.py       the walk-through above
  engine.py             shared by the pages: reading uploads, learning a person, predicting, charts, notes
  style.py              colours (Flexoki, light and dark), fonts, the logo
models/
  parse.py              reads CSV / Excel / JSON exports in many shapes into one format
  clean.py              smooths sensor noise
  classifier.py         type 1 or not
  metabolic.py, t1d.py  load the two datasets (CGMacros; MetaboNet type 1)
  context.py, events.py what the model knows at a meal (time of day, meals so far, exercise, ...)
  exogenous.py          food and insulin -> curves in the blood (OpenAPS)
  regressor.py          model 1: the regression
  overlay.py            model 2: S-curves per event
  common.py             fitting everyone -> group -> person, and the band
  train.py              trains and scores both models  ->  models/weights/
  photo2nutrition.py    photo -> nutrition (CLIP + USDA FNDDS + photo memory)
  plan.py               a planned meal, exercise or dose -> the inputs the model reads
samples/                demo people and sample files to drop into the site
results/                scores, and the numbers How it works shows
docs/                   the project page (GitHub Pages)
```

## Credits

Data: CGMacros (PhysioNet) and MetaboNet (type 1 studies). Food: USDA FNDDS 2021-2023 (public domain). Photo model:
OpenAI CLIP ViT-L/14 (MIT), run with OpenVINO where available. Insulin curve: OpenAPS. Font: Libron (SIL OFL 1.1).
Colours: Flexoki. Pronunciation: Jeuwre, CC BY-SA 4.0, Wikimedia Commons. Sensor photos and anatomy plates:
see `static/hiw/CREDITS.md`.

Not medical advice.
