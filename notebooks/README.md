# Notebooks

Thin, ordered drivers for the analysis. All heavy logic is imported from the
`src` package (and unit-tested), so notebooks stay readable and reproducible.

Run order:

1. **eda-fraud-data.ipynb** — Univariate/bivariate EDA and class imbalance for the
   e-commerce dataset.
2. **eda-creditcard.ipynb** — EDA and (severe) class imbalance for the bank dataset.
3. **feature-engineering.ipynb** — IP→country geolocation, temporal + velocity
   features, scaling/encoding, and the SMOTE demonstration. Writes both processed
   CSVs to `data/processed/`.
4. **modeling.ipynb** — Logistic Regression baseline vs XGBoost/Random-Forest
   ensemble, AUC-PR / F1 / confusion matrix, 5-fold CV, and model selection.
5. **shap-explainability.ipynb** — built-in importance, SHAP global summary,
   force/waterfall plots for TP/FP/FN, and business recommendations.

> Notebooks expect the raw CSVs in `data/raw/`. For a dry run without real data,
> first run `python scripts/make_synthetic_data.py`.

The notebooks are generated from `scripts/build_notebooks.py`; edit that file and
re-run it to regenerate them consistently.
