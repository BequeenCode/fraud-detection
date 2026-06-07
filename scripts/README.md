# Scripts

Command-line drivers for the project pipeline. Run from the repository root.

| Script | Purpose |
|--------|---------|
| `make_synthetic_data.py` | Generate small **synthetic** datasets (same schema as the real files) into `data/raw/` so the pipeline can run without the real data. **For code testing only.** |
| `run_preprocessing.py` | **Task 1** — clean → geolocate (IP→country) → engineer features for both datasets; writes `data/processed/*.csv`. |
| `run_modeling.py` | **Task 2** — stratified split, train Logistic Regression baseline + XGBoost/Random-Forest ensemble, evaluate (AUC-PR, F1, confusion matrix), save the best model to `models/`. |
| `build_notebooks.py` | Regenerate the five notebooks in `notebooks/` from a single reviewable source. |

Typical flow:

```bash
python scripts/run_preprocessing.py   # produces data/processed/
python scripts/run_modeling.py        # produces models/ + metrics tables
```
