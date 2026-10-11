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

## Layout

| Folder | What's in it |
|---|---|
| `views/` | the site's pages |
| `models/` | data reading, the models, training, photo to nutrition; `models/weights/` the trained models |
| `samples/` | demo people and sample sets you can drop into the site |
| `static/` | fonts, art, audio, small copies of logged meal photos |
| `results/` | scores and the numbers behind How it works |
| `docs/` | the project page (GitHub Pages) |

## Credits

Data: CGMacros (PhysioNet) and MetaboNet (type 1 studies). Food: USDA FNDDS 2021-2023 (public domain). Photo model:
OpenAI CLIP ViT-L/14 (MIT), run with OpenVINO where available. Insulin curve: OpenAPS. Font: Libron (SIL OFL 1.1).
Colours: Flexoki. Pronunciation: Jeuwre, CC BY-SA 4.0, Wikimedia Commons. Sensor photos and anatomy plates:
see `static/hiw/CREDITS.md`.

Not medical advice.
