# Interim-1 Report — Task 1: Data Analysis & Preprocessing

**Project:** Fraud Detection for E-commerce and Bank Transactions
**Company (case):** Adey Innovations Inc.
**Scope of this report:** Data cleaning, EDA, geolocation integration, feature
engineering, and class-imbalance strategy for the two target datasets.

> Reproducibility: every step below is implemented as tested functions in `src/`
> and executed end-to-end by `scripts/run_preprocessing.py`. Figures referenced
> here are written to `reports/figures/` when the notebooks are run on the real data.

---

## 1. Datasets & framing

| Dataset | Rows (typical) | Target | Fraud share |
|---------|---------------:|--------|------------:|
| `Fraud_Data.csv` (e-commerce) | ~151k | `class` | ≈ **9.4%** |
| `creditcard.csv` (bank) | ~284k | `Class` | ≈ **0.17%** |
| `IpAddress_to_Country.csv` | ~138k ranges | — | — |

Both targets are **highly imbalanced**. The business cost is asymmetric: a
**false negative** (missed fraud) is a direct financial loss, while a **false
positive** (legit flagged) erodes customer trust. This shapes two decisions that
run through the whole project:

1. **Metrics:** accuracy is misleading on imbalanced data (a model predicting
   "never fraud" scores 99.8% on the credit-card set). We lead with **AUC-PR
   (average precision)**, **F1**, and the **confusion matrix**, with ROC-AUC as a
   secondary view.
2. **Resampling:** handled on the **training set only** to avoid leakage.

---

## 2. Data cleaning

Implemented in `src/cleaning.py` (pure functions, unit-tested in
`tests/test_cleaning.py`).

### E-commerce (`Fraud_Data.csv`)
- **Duplicates:** exact duplicate rows removed (`drop_duplicates`).
- **Data types:** `signup_time` / `purchase_time` parsed to `datetime`;
  `ip_address` coerced to numeric (stored as float in the raw file);
  `source`, `browser`, `sex` cast to `category`.
- **Missing values:** rows missing an *essential* field (`signup_time`,
  `purchase_time`, `ip_address`) are **dropped** — they cannot be
  feature-engineered or geolocated. Non-essential numeric/categorical gaps
  (e.g. occasional `age`) are **imputed inside the modelling pipeline**
  (median for numeric, most-frequent for categorical) so imputation statistics
  are fit on the training fold only.

### Bank (`creditcard.csv`)
- No missing values, but the file contains a meaningful number of **exact
  duplicate rows**; these are removed to prevent identical records leaking
  across the train/test split.
- `V1..V28` are already standardised PCA outputs; `Time` and `Amount` are scaled
  in the modelling pipeline.

### Justification of missing-value handling
We **drop** only when a record is structurally unusable (no timestamp / no IP),
which affects a tiny fraction of rows and avoids fabricating the very signals
(time-since-signup, geolocation) we rely on. We **impute** everything else inside
the pipeline rather than up front, which prevents test-set statistics from
leaking into preprocessing.

---

## 3. Exploratory Data Analysis

(See `notebooks/eda-fraud-data.ipynb` and `notebooks/eda-creditcard.ipynb`.)

### Class imbalance (quantified)
- E-commerce: ~9% positive — moderate imbalance.
- Credit-card: ~0.17% positive — severe imbalance (≈ 1 fraud per ~580 legit).
  Plotted on a log scale to be legible.

### Univariate (e-commerce)
- `purchase_value` is right-skewed; `age` is roughly 18–70 with a central mode.
- Channel mix across `source` (SEO / Ads / Direct) and `browser`.

### Bivariate (feature ↔ target)
- **Fraud rate by `source` / `browser` / `sex`** — fraud propensity varies by
  acquisition channel and browser, making them useful categorical signals.
- **`purchase_value` by class** — distributional differences inspected via boxplot.
- **Credit-card:** a handful of PCA components (ranked by
  |mean(fraud) − mean(legit)|) carry most of the separating signal; `Amount`
  differs by class.

**EDA conclusion:** the strongest expected predictors are the **engineered
temporal/velocity features** (Section 5), not the raw columns alone.

---

## 4. Geolocation integration (IP → country)

Implemented in `src/geolocation.py`; range-lookup correctness covered by
`tests/test_geolocation.py`.

1. **IP → integer.** `ip_to_int` handles both encodings: numeric floats in the
   raw file (truncated to int) and dotted-quad strings
   (`a·256³ + b·256² + c·256 + d`).
2. **Range-based merge.** The lookup gives countries for *ranges*, so an equality
   join fails. We:
   - sort the lookup by `lower_bound_ip_address`,
   - use `pandas.merge_asof(direction="backward")` — a vectorised binary search
     that finds, per transaction, the range whose lower bound is the greatest
     value ≤ the IP,
   - **validate the upper bound**: if `ip_int > upper_bound`, the IP fell in a
     gap between ranges and is labelled `"Unknown"`.
   - Original row order is preserved (sort → merge → restore).

   This is **O(n log n)** and processes the full dataset in well under a second.
3. **Fraud by country.** `fraud_rate_by_country` reports transaction volume and
   fraud rate per country (restricted to countries with ≥ 50 transactions so
   genuinely risky geographies surface ahead of low-volume noise).

---

## 5. Feature engineering (e-commerce)

Implemented in `src/feature_engineering.py`; covered by
`tests/test_feature_engineering.py`.

| Feature | Definition | Fraud rationale |
|---------|-----------|-----------------|
| **`time_since_signup`** | seconds & hours between `signup_time` and `purchase_time` | Fraud rings often purchase within seconds/minutes of creating an account; legit users wait far longer. **Expected top predictor.** |
| `hour_of_day` | hour of `purchase_time` | Captures off-hours fraud patterns. |
| `day_of_week` | weekday of purchase (Mon=0) | Weekly seasonality. |
| `user_tx_count` | total purchases per user | Repeat-use signal. |
| `device_tx_count` | total purchases per device | High counts hint at automation. |
| `device_shared_users` | distinct users per device | **Many users on one device** is a classic fraud-ring signal. |
| `sec_since_prev_user_tx` | seconds since the user's previous purchase | Rapid-fire "velocity" bursts. |

### Spotlight: `time_since_signup`
Computed as `(purchase_time − signup_time)` in seconds (and hours). In EDA the
**median time-since-signup for fraud is dramatically smaller** than for
legitimate transactions, which is why it is engineered explicitly and expected to
dominate both built-in and SHAP importance in later tasks.

---

## 6. Data transformation

Implemented in `src/preprocessing.py` as a scikit-learn `ColumnTransformer`,
**fit on training data only** inside the model pipeline:

- **Numeric:** `SimpleImputer(median)` → `StandardScaler`.
- **Categorical** (`source`, `browser`, `sex`, `country`):
  `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`.
- Identifier / raw columns (`user_id`, `device_id`, timestamps, `ip_address`,
  `ip_int`) are dropped from the model matrix.

For the credit-card data, all features (`V1..V28`, `Time`, `Amount`) are
imputed (median) and scaled.

---

## 7. Class-imbalance strategy

| Aspect | Decision |
|--------|----------|
| **Primary technique** | **SMOTE** (synthetic minority oversampling) on the **training fold only**. |
| **Why SMOTE over undersampling** | Fraud is rare in *absolute* terms; random undersampling would discard the large majority of legitimate transactions and lose information. SMOTE preserves all majority data and synthesises plausible minority examples. |
| **Leakage control** | Applied **inside** an `imblearn` pipeline so it runs within each CV fold and never touches validation/test data. |
| **Model-level alternative** | For the extreme credit-card imbalance we also use built-in weighting — `class_weight="balanced"` (Logistic Regression / Random Forest) and `scale_pos_weight = n_neg/n_pos` (XGBoost) — which is often more stable than oversampling at 0.17% prevalence. |

### Documented distribution (illustrative, e-commerce training fold)
```
before SMOTE:  [ majority≈91% , minority≈9%  ]
after  SMOTE:  [ majority 50% , minority 50% ]   # minority synthesised up to parity
```
The exact counts are printed by the feature-engineering notebook and
`scripts/run_preprocessing.py` on the real data.

---

## 8. Deliverables status (Interim-1)

- [x] Cleaned datasets — `scripts/run_preprocessing.py` → `data/processed/`.
- [x] EDA report — this document + the two EDA notebooks.
- [x] Feature-engineering documentation — Section 5 + `src/feature_engineering.py`.
- [x] Geolocation (IP→country) — Section 4 + `src/geolocation.py`.
- [x] Resampling justification — Section 7.
- [x] Reproducible, tested code — `src/` + `tests/` (14 unit tests) + CI.

### What's next (Interim-2, Task 2)
Train and compare the Logistic Regression baseline against an XGBoost/Random-Forest
ensemble using AUC-PR / F1 / confusion matrix and 5-fold stratified CV, then select
and persist the best model — already scaffolded in `notebooks/modeling.ipynb` and
`scripts/run_modeling.py`.
