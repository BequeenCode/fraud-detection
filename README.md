# Fraud Detection for E-commerce and Bank Transactions

Improved fraud detection for **Adey Innovations Inc.** across two transaction
streams — rich-context e-commerce transactions and PCA-anonymised bank
credit-card transactions. The project covers the full workflow: data cleaning,
geolocation enrichment, feature engineering, imbalance-aware modelling, and SHAP
explainability tied back to business recommendations.

## Datasets

The three source files are **not committed** (see `.gitignore`). Download them and
place them in `data/raw/`:

| File | Description |
|------|-------------|
| `Fraud_Data.csv` | E-commerce transactions (`signup_time`, `purchase_time`, `device_id`, `ip_address`, `class`, …) |
| `IpAddress_to_Country.csv` | IP-range → country lookup table |
| `creditcard.csv` | Bank transactions (`Time`, `V1..V28` PCA features, `Amount`, `Class`) |

Both target datasets are **highly imbalanced** (e-commerce ≈ 9% fraud,
credit-card ≈ 0.17% fraud), which drives the choice of metrics (AUC-PR, F1,
confusion matrix — **not** accuracy) and resampling strategy (SMOTE on the
training set only).

## Project structure

```
fraud-detection/
├── data/                # raw/ + processed/ (gitignored)
├── notebooks/           # EDA, feature engineering, modeling, SHAP
├── src/                 # reusable, tested library code (the real logic)
├── scripts/             # end-to-end pipeline runners + helpers
├── tests/               # pytest unit tests (run in CI)
├── models/              # saved model artifacts (gitignored)
├── reports/             # Interim report + generated figures
├── requirements.txt
└── .github/workflows/   # CI: runs the test suite
```

The design principle: **all analytical logic lives in `src/`** and is unit-tested;
notebooks and scripts are thin drivers that call it, so the same code runs in
notebooks, the CLI pipeline, and CI.

## Quickstart

```bash
# 1. Install dependencies (a virtualenv is recommended)
pip install -r requirements.txt

# 2. Put the three CSVs in data/raw/

# 3. Task 1 — clean, geolocate, feature-engineer -> data/processed/
python scripts/run_preprocessing.py

# 4. Task 2 — train + evaluate baseline and ensemble -> models/
python scripts/run_modeling.py

# 5. Run the tests
pytest -q
```

### Notebooks

Run them in this order (each is self-contained and imports from `src`):

1. `notebooks/eda-fraud-data.ipynb` — e-commerce EDA
2. `notebooks/eda-creditcard.ipynb` — credit-card EDA
3. `notebooks/feature-engineering.ipynb` — geolocation, features, scaling, SMOTE → writes `data/processed/`
4. `notebooks/modeling.ipynb` — baseline vs ensemble, metrics, CV, model selection
5. `notebooks/shap-explainability.ipynb` — importance, SHAP summary, force plots, recommendations

> Don't have the data yet? `python scripts/make_synthetic_data.py` writes small
> **synthetic** files with the same schema so you can exercise the whole pipeline.
> The numbers are fabricated — for code testing only, never for analysis.

## Highlights of the approach

- **Geolocation by IP range** — IPs are converted to integers and matched to
  country ranges with a vectorised `merge_asof` + upper-bound validation
  (`src/geolocation.py`), O(n log n) over the full dataset.
- **Behavioural features** — `time_since_signup`, `hour_of_day`, `day_of_week`,
  and velocity signals (`user_tx_count`, `device_shared_users`,
  `sec_since_prev_user_tx`).
- **Imbalance handling** — SMOTE on the training fold only, plus
  `class_weight` / `scale_pos_weight` model options; metrics centred on AUC-PR.
- **Explainability** — built-in importance vs SHAP, with global summary and
  per-prediction force/waterfall plots, translated into concrete fraud-rules.

## Reports

- [`reports/interim1_report.md`](reports/interim1_report.md) — Task 1 write-up
  (cleaning, EDA, feature engineering, IP-to-country mapping, imbalance strategy).

## Tech stack

Python 3.11+, pandas, scikit-learn, imbalanced-learn, XGBoost, SHAP, matplotlib/seaborn.
