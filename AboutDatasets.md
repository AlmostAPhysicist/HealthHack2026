# About the Datasets

## The challenge in short

- Input: 7 days of one person's data (CGM, heart rate, meals + photos, medication).
- Output: their glucose curve after a new meal on Day 8.
- Predict: glucose at 30/60/90/120 min, peak glucose, rise from pre-meal to peak, time to peak, 2-hour iAUC.
- Judged on: accuracy, personalization (not just a population average), explaining why, using all data types, and usability.
- Note: the challenge says the judging data is **synthetic**. The two datasets here are real. Use them to build and test the approach.

## Side by side

| | **CGMacros** (`data/cgmacros/`) | **Type1Diabetes.parquet** (MetaboNet) |
|---|---|---|
| Who | 45 adults: healthy, prediabetic, type 2 | ~1,400 type 1 diabetics from 15 studies |
| Size | 45 CSVs, ~10 days each | 1.3 GB, 133M rows |
| Time step | 1 min | 5 min |
| Glucose | 2 CGMs per person (Libre + Dexcom) | 1 `CGM` column |
| Meals | Calories, carbs, protein, fat, fiber + **photos** | Carbs only. Meal names are very rare |
| Heart rate | Yes, for everyone | Only 4 small studies |
| Activity | Calories burned, METs | Steps/calories in a few studies |
| Medication | **None** | Insulin (basal, bolus, total) + pump/device info |
| Per-person extras | Blood labs (A1c, fasting glucose/insulin, lipids), gut health, microbes | Age, gender, height, weight, age at diagnosis |

## Which one to use

- **CGMacros is the main one.** It is the only one with meal photos and full macros, which is the core of the Day 8 task. Build the per-person model here.
- **The parquet is backup for medication.** CGMacros has no medication. The parquet has insulin dosing, so use it to learn how insulin timing and dose change the post-meal curve.
- Optional: pre-train a general model on the parquet (lots of people), then fine-tune per person on CGMacros. Different diseases though (type 1 vs type 2/healthy), so the transfer is not clean.

---

## 1. CGMacros

From PhysioNet. Dates are shifted, so treat them as relative.

### Files

```
data/cgmacros/CGMacros_dateshifted365/CGMacros/
├── CGMacros-0XX/            45 people (001 to 049, missing 024, 025, 037, 040)
│   ├── CGMacros-0XX.csv     time series, 1 row per minute
│   └── photos/              meal photos (~90 per person)
├── bio.csv                  1 row per person: demographics + blood labs
├── gut_health_test.csv      1 row per person: gut pathway scores (1 to 3)
└── microbes.csv             1 row per person: microbe present (1) / absent (0)
```

Dictionaries: `data/cgmacros/DataDictionary_*.csv` and `DataDictionary.pdf`.

### Time series columns

| Column | What it is |
|---|---|
| `Timestamp` | Every minute |
| `Libre GL` | Glucose, Libre Pro sensor (mg/dL, 40 to 400) |
| `Dexcom GL` | Glucose, Dexcom G6 sensor (mg/dL) |
| `HR` | Heart rate |
| `Calories (Activity)`, `METs` | Activity level |
| `Meal Type` | Breakfast / Lunch / Dinner / Snacks. Only filled on meal rows |
| `Calories`, `Carbs`, `Protein`, `Fat`, `Fiber` | What was in the meal |
| `Amount Consumed` | % of the meal eaten |
| `Image path` | Photo file, e.g. `photos/00000005-PHOTO-...jpg`. Some meals have none |

### Per-person tables

- `bio.csv`: age, gender, BMI, weight, height, ethnicity, A1c, fasting glucose, fasting insulin, cholesterol panel, 3 fingerstick readings. A1c tells you healthy vs prediabetic vs T2D. LDL = 800 and VLDL = 400 are error codes.
- `gut_health_test.csv`: ~20 gut scores like butyrate production, metabolic fitness.
- `microbes.csv`: huge binary table. Too many columns for 45 people. Shrink it (PCA, pick a few) or skip it.

### What you can do with it

- Find each meal, cut out the glucose from 0 to 120 min after it. That gives the targets (30/60/90/120 min, peak, time to peak, iAUC).
- Features per meal: macros, pre-meal glucose, glucose trend over the last 30 to 60 min, HR and activity before the meal, time of day, time since last meal.
- Split each person: first 7 days train, later meals test. Same setup as the challenge.
- Photos: train or use a vision model (or an LLM) to estimate macros from the photo, check against the true macros.
- Personalization: fit one shared model plus a per-person adjustment (e.g. their own carb sensitivity). Compare against a single model for everyone.
- Use `bio.csv` (A1c, fasting insulin) to explain why people differ.
- Two CGMs on one person shows how noisy the sensors are. Useful for uncertainty.

---

## 2. Type1Diabetes.parquet (MetaboNet)

A merge of 15 public type 1 diabetes studies into one table. Units and column names are made consistent and everything is resampled to a 5-minute grid. One row = one person at one 5-minute step. Dictionary: https://metabo-net.org/data-dictionary

Explore it with [peakParquetDataset.ipynb](notebooks/peakParquetDataset.ipynb). Query with DuckDB. Do not load it all into pandas.

### Columns

"Last 5 min" means the value is for the 5 minutes before that row's timestamp.

**Core**

| Column | Unit | What it is |
|---|---|---|
| `id` | | Person ID (string). Use `(source_file, id)` as the key |
| `date` | | Timestamp |
| `source_file` | | Which study the row came from |
| `CGM` | mg/dL | Glucose reading |
| `basal` | units | Background insulin, last 5 min. IOBP2 has none (all logged as bolus) |
| `bolus` | units | Mealtime/correction insulin, last 5 min |
| `insulin` | units | basal + bolus |
| `carbs` | g | Carbs the person logged. Mix of 0s and nulls depending on study. 3 to 4 per day max |
| `meal_label` | | Meal name in free text. Very rare (0.07% of rows) |

**Wearables (few studies only)**

| Column | Unit | What it is |
|---|---|---|
| `heartrate` | bpm | Average HR, last 5 min |
| `steps` | count | Steps, last 5 min |
| `calories_burned` | kcal | Calories, last 5 min |
| `workout_duration` | min | Exercise length |
| `workout_intensity` | 1 to 10 | Self-rated. Almost always empty |
| `workout_label` | | e.g. Running, Cycling. Almost always empty |
| `galvanic_skin_response` | µS | Sweat/stress signal. OhioT1DM only |
| `skin_temp`, `air_temp` | °F | OhioT1DM only |

**Person info (same value on every row for a person)**

| Column | Unit | What it is |
|---|---|---|
| `age` | years | |
| `gender` | | Female / Male / Non-binary |
| `height` | **feet** | |
| `weight` | **lbs** | |
| `age_of_diagnosis` | years | Age when diagnosed with T1D |
| `ethnicity` | | Comma separated if more than one |
| `is_pregnant` | | At start of study |

**Devices**

| Column | What it is |
|---|---|
| `cgm_device` | e.g. Dexcom G6, Libre |
| `insulin_delivery_modality` | `MDI` = injections, `SAP` = pump, `AID` = automated pump (closed loop) |
| `insulin_delivery_device` | e.g. Omnipod, MiniMed 670G |
| `insulin_delivery_algorithm` | Closed-loop software, e.g. Loop, Control-IQ |
| `insulin_type_basal`, `insulin_type_bolus` | e.g. humalog, fiasp |

**Study bookkeeping (mostly ignore)**

`treatment_group`, `randomization_date`, `extension_date` (clinical trial info), `misc_notes`, `is_test`, `subject_split_across_traintest`.

### Which studies have what

| Study | People | Rows | % CGM | % carbs | % HR | % steps |
|---|---:|---:|---:|---:|---:|---:|
| Loop | 761 | 81.0M | 89 | 83 | 0 | 0 |
| ReplaceBG | 188 | 18.1M | 64 | 88 | 0 | 0 |
| IOBP2 | 299 | 8.3M | 80 | 0 | 0 | 0 |
| Flair | 102 | 6.0M | 79 | 0 | 0 | 0 |
| DCLP5 | 90 | 5.2M | 90 | 0 | 0 | 0 |
| DCLP3 | 101 | 5.2M | 93 | 0 | 0 | 0 |
| PEDAP | 59 | 4.7M | 91 | 100 | 0 | 0 |
| CTR3 | 30 | 2.0M | 57 | 1 | 0 | 0 |
| BrisT1D | 18 | 973k | 75 | 2 | 59 | 41 |
| T1D-UOM | 13 | 652k | 36 | 1 | 0 | 16 |
| AZT1D | 21 | 242k | 98 | 7 | 0 | 0 |
| HUPA-UCM | 20 | 207k | 100 | 1 | 100 | 100 |
| OhioT1DM | 12 | 191k | 87 | 94 | 46 | 48 |
| ShanghaiT1DM | 11 | 42k | 33 | 0 | 0 | 0 |
| SIESTA | 5 | 38k | 97 | 0 | 84 | 0 |

"% carbs" counts non-null rows, and most of those are 0. Real meals are rare.

Insulin setup across people: 1,114 AID, 226 SAP, 24 MDI, 20 unknown.

### Watch out for

- **Find meals with `carbs > 0`**, not "carbs is not null". ~87M rows are 0.
- **HR + carbs together:** only OhioT1DM really has both. HUPA-UCM and BrisT1D have HR but few carb logs.
- **No photos, no protein/fat/fiber.** Useless for the photo part.
- **Closed-loop pumps (AID) react to glucose.** High glucose causes more insulin, so insulin can look like it "causes" high glucose. Be careful when explaining feature importance.
- `is_test` is false everywhere. Make your own split.
- Dates are shifted (OhioT1DM goes into 2027).
- Carbs are self-reported, so they are noisy.

### What you can do with it

- Learn how insulin dose and timing before a meal change the glucose curve. This covers the "medication" part CGMacros lacks.
- Big population model: lots of people, so you can measure how much of the response is personal vs shared.
- OhioT1DM is the closest match to the challenge (CGM + carbs + insulin + HR + steps). Good place to test the full pipeline with medication.
- Find people with few meals and see how the model's confidence drops. Useful for the uncertainty part.

---

## Quick start

```python
import duckdb
con = duckdb.connect()

# Parquet: real meals in OhioT1DM
con.sql("""
    SELECT id, date, CGM, carbs, bolus, heartrate
    FROM 'data/Type1Diabetes.parquet'
    WHERE source_file = 'OhioT1DM' AND carbs > 0
    LIMIT 10
""").df()

# CGMacros: all meals for subject 001
con.sql("""
    SELECT * FROM read_csv_auto('data/cgmacros/CGMacros_dateshifted365/CGMacros/CGMacros-001/CGMacros-001.csv')
    WHERE "Meal Type" IS NOT NULL
""").df()
```
