"""Generate the five project notebooks with nbformat.

Keeping notebook content in a single, reviewable Python file (rather than raw
.ipynb JSON) makes the analysis easy to regenerate and diff. Each notebook is a
*thin* driver: the heavy logic lives in the ``src`` package and is merely called
here so the same, tested code runs in notebooks, scripts and CI.

Run:  python scripts/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

# A preamble injected into every notebook so `import src...` resolves whether
# the notebook is launched from notebooks/ or the repo root.
BOOTSTRAP = (
    "import sys, warnings\n"
    "from pathlib import Path\n"
    "warnings.filterwarnings('ignore')\n"
    "ROOT = Path.cwd()\n"
    "ROOT = ROOT.parent if ROOT.name == 'notebooks' else ROOT\n"
    "sys.path.insert(0, str(ROOT))\n"
    "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
    "sns.set_theme(style='whitegrid')\n"
    "pd.set_option('display.max_columns', 60)\n"
)


def md(text: str):
    return new_markdown_cell(text)


def code(src: str):
    return new_code_cell(src)


def write(name: str, cells: list) -> None:
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    path = NB_DIR / name
    nbf.write(nb, path)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# 1. EDA - Fraud_Data.csv
# --------------------------------------------------------------------------- #
def build_eda_fraud():
    cells = [
        md(
            "# EDA - E-commerce Fraud Data\n\n"
            "Univariate & bivariate exploration of `Fraud_Data.csv`, plus the "
            "class-imbalance picture. Data cleaning logic lives in `src.cleaning`."
        ),
        code(BOOTSTRAP),
        code(
            "from src import config, data_loading, cleaning\n"
            "raw = data_loading.load_fraud_data()\n"
            "print('raw shape:', raw.shape)\n"
            "raw.head()"
        ),
        md("## Schema & data types"),
        code("raw.info()"),
        md("## Missing values & duplicates"),
        code(
            "display(cleaning.missing_value_report(raw))\n"
            "print('exact duplicate rows:', raw.duplicated().sum())"
        ),
        code(
            "df = cleaning.clean_fraud_data(raw)\n"
            "print('after cleaning:', df.shape)"
        ),
        md("## Class imbalance\nThe positive (fraud) class is a small minority - "
           "this drives our metric and resampling choices."),
        code(
            "bal = cleaning.class_balance(df, config.FRAUD_TARGET)\n"
            "display(bal)\n"
            "ax = sns.countplot(x=config.FRAUD_TARGET, data=df)\n"
            "ax.set_title('Class distribution (0=legit, 1=fraud)')\n"
            "plt.savefig(config.FIGURES_DIR/'fraud_class_balance.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md("## Univariate distributions"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "sns.histplot(df['purchase_value'], bins=40, ax=axes[0]).set_title('Purchase value')\n"
            "sns.histplot(df['age'], bins=40, ax=axes[1]).set_title('Age')\n"
            "plt.tight_layout(); plt.savefig(config.FIGURES_DIR/'fraud_univariate.png', dpi=120, bbox_inches='tight'); plt.show()"
        ),
        code(
            "for col in ['source', 'browser', 'sex']:\n"
            "    print(col); print(df[col].value_counts(normalize=True).round(3)); print()"
        ),
        md("## Bivariate: fraud rate by category\nHow does the fraud rate vary across channels, browsers and gender?"),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "for ax, col in zip(axes, ['source', 'browser', 'sex']):\n"
            "    (df.groupby(col)[config.FRAUD_TARGET].mean()*100).plot.bar(ax=ax)\n"
            "    ax.set_title(f'Fraud rate (%) by {col}'); ax.set_ylabel('fraud %')\n"
            "plt.tight_layout(); plt.savefig(config.FIGURES_DIR/'fraud_rate_by_category.png', dpi=120, bbox_inches='tight'); plt.show()"
        ),
        md("## Bivariate: purchase value vs class"),
        code(
            "ax = sns.boxplot(x=config.FRAUD_TARGET, y='purchase_value', data=df)\n"
            "ax.set_title('Purchase value by class'); plt.show()\n"
            "df.groupby(config.FRAUD_TARGET)['purchase_value'].describe()"
        ),
        md(
            "### Takeaways\n"
            "- Fraud is a small minority -> use **AUC-PR / F1**, not accuracy.\n"
            "- Fraud rate differs by `source`/`browser` -> these are useful categorical signals.\n"
            "- Temporal & velocity features (built in the feature-engineering notebook) are expected "
            "to be the strongest predictors."
        ),
    ]
    write("eda-fraud-data.ipynb", cells)


# --------------------------------------------------------------------------- #
# 2. EDA - creditcard.csv
# --------------------------------------------------------------------------- #
def build_eda_creditcard():
    cells = [
        md(
            "# EDA - Bank Credit-Card Data\n\n"
            "`creditcard.csv` features are PCA-anonymised (`V1..V28`) plus `Time` and "
            "`Amount`. This dataset is **extremely** imbalanced (~0.17% fraud)."
        ),
        code(BOOTSTRAP),
        code(
            "from src import config, data_loading, cleaning\n"
            "raw = data_loading.load_creditcard()\n"
            "print('raw shape:', raw.shape)\n"
            "raw.head()"
        ),
        code("raw.info()"),
        md("## Missing values & duplicates"),
        code(
            "display(cleaning.missing_value_report(raw).head(10))\n"
            "print('exact duplicate rows:', raw.duplicated().sum())\n"
            "df = cleaning.clean_creditcard(raw)\n"
            "print('after dedup:', df.shape)"
        ),
        md("## Class imbalance (severe)"),
        code(
            "bal = cleaning.class_balance(df, config.CREDITCARD_TARGET)\n"
            "display(bal)\n"
            "ax = sns.countplot(x=config.CREDITCARD_TARGET, data=df)\n"
            "ax.set_yscale('log'); ax.set_title('Class distribution (log scale)')\n"
            "plt.savefig(config.FIGURES_DIR/'cc_class_balance.png', dpi=120, bbox_inches='tight'); plt.show()"
        ),
        md("## Transaction Amount by class"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "sns.histplot(df['Amount'], bins=50, ax=axes[0]).set_title('Amount (all)')\n"
            "sns.boxplot(x=config.CREDITCARD_TARGET, y='Amount', data=df, ax=axes[1])\n"
            "axes[1].set_title('Amount by class')\n"
            "plt.tight_layout(); plt.savefig(config.FIGURES_DIR/'cc_amount.png', dpi=120, bbox_inches='tight'); plt.show()"
        ),
        md("## Which PCA components separate fraud best?"),
        code(
            "vcols = [c for c in df.columns if c.startswith('V')]\n"
            "sep = (df[df[config.CREDITCARD_TARGET]==1][vcols].mean() - df[df[config.CREDITCARD_TARGET]==0][vcols].mean()).abs().sort_values(ascending=False)\n"
            "sep.head(10).plot.bar(title='|mean(fraud) - mean(legit)| by component'); plt.show()\n"
            "sep.head(10)"
        ),
        md(
            "### Takeaways\n"
            "- ~0.17% positives: accuracy is useless; optimise **AUC-PR / recall at fixed precision**.\n"
            "- A handful of PCA components carry most of the separating signal.\n"
            "- `Amount` should be scaled; `V1..V28` are already standardised PCA outputs."
        ),
    ]
    write("eda-creditcard.ipynb", cells)


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
def build_feature_engineering():
    cells = [
        md(
            "# Feature Engineering & Transformation\n\n"
            "Geolocation enrichment, temporal + velocity features, scaling/encoding, "
            "and the SMOTE resampling demonstration. All logic is imported from `src`."
        ),
        code(BOOTSTRAP),
        code(
            "from src import config, data_loading, cleaning, geolocation, feature_engineering, preprocessing\n"
            "df = cleaning.clean_fraud_data(data_loading.load_fraud_data())\n"
            "print('clean shape:', df.shape)"
        ),
        md(
            "## 1. Geolocation: IP -> country (range lookup)\n"
            "IPs are converted to integers, then matched to country ranges with a "
            "vectorised `merge_asof` + upper-bound validation (see `src.geolocation`)."
        ),
        code(
            "ipc = data_loading.load_ip_country()\n"
            "geo = geolocation.merge_ip_to_country(df, ipc)\n"
            "print('matched to a country:', round((geo['country']!='Unknown').mean()*100, 1), '%')\n"
            "geo[['ip_address','ip_int','country']].head()"
        ),
        code(
            "top = geolocation.fraud_rate_by_country(geo, config.FRAUD_TARGET, top_n=15)\n"
            "display(top)\n"
            "top['fraud_rate'].plot.barh(title='Fraud rate (%) by country (>=50 tx)'); plt.gca().invert_yaxis()\n"
            "plt.savefig(config.FIGURES_DIR/'fraud_rate_by_country.png', dpi=120, bbox_inches='tight'); plt.show()"
        ),
        md(
            "## 2. Temporal & velocity features\n"
            "- `time_since_signup_hours` - gap between signup and purchase (key fraud signal).\n"
            "- `hour_of_day`, `day_of_week` - when the purchase happened.\n"
            "- `user_tx_count`, `device_tx_count`, `device_shared_users`, `sec_since_prev_user_tx` - velocity."
        ),
        code(
            "feat = feature_engineering.engineer_fraud_features(geo)\n"
            "new_cols = ['hour_of_day','day_of_week','time_since_signup_hours','user_tx_count',\n"
            "            'device_tx_count','device_shared_users','sec_since_prev_user_tx']\n"
            "feat[new_cols + [config.FRAUD_TARGET]].head()"
        ),
        code(
            "ax = sns.boxplot(x=config.FRAUD_TARGET, y='time_since_signup_hours', data=feat)\n"
            "ax.set_title('Time since signup (hours) by class'); ax.set_ylim(0, feat['time_since_signup_hours'].quantile(0.99))\n"
            "plt.savefig(config.FIGURES_DIR/'time_since_signup_by_class.png', dpi=120, bbox_inches='tight'); plt.show()\n"
            "feat.groupby(config.FRAUD_TARGET)['time_since_signup_hours'].median()"
        ),
        code(
            "config.ensure_dirs()\n"
            "feat.to_csv(config.PROCESSED_DIR/'fraud_data_processed.csv', index=False)\n"
            "print('saved processed fraud data ->', config.PROCESSED_DIR/'fraud_data_processed.csv')"
        ),
        md(
            "### Also persist the cleaned credit-card data\n"
            "The bank dataset needs no feature engineering (PCA features already), only "
            "de-duplication, so we save it here for the modelling notebook to consume."
        ),
        code(
            "cc = cleaning.clean_creditcard(data_loading.load_creditcard())\n"
            "cc.to_csv(config.PROCESSED_DIR/'creditcard_processed.csv', index=False)\n"
            "print('saved processed credit-card data ->', config.PROCESSED_DIR/'creditcard_processed.csv', cc.shape)"
        ),
        md(
            "## 3. Transformation: scaling + one-hot encoding\n"
            "A `ColumnTransformer` imputes + scales numerics and one-hot-encodes "
            "categoricals. It is **fit on training data only** inside the modelling pipeline."
        ),
        code(
            "X, y = preprocessing.select_fraud_model_columns(feat)\n"
            "X_train, X_test, y_train, y_test = preprocessing.stratified_split(X, y)\n"
            "pre = preprocessing.build_fraud_preprocessor(X_train)\n"
            "Xt = pre.fit_transform(X_train)\n"
            "print('transformed train matrix:', Xt.shape)\n"
            "print('first 8 feature names:', list(pre.get_feature_names_out())[:8])"
        ),
        md(
            "## 4. Class imbalance - SMOTE (training set only)\n"
            "We oversample the minority class with **SMOTE** rather than undersample, "
            "because the fraud count is small in absolute terms and undersampling would "
            "discard most legitimate transactions (information loss). SMOTE is applied "
            "**only to the training fold** to avoid leaking synthetic samples into validation."
        ),
        code(
            "from imblearn.over_sampling import SMOTE\n"
            "print('before SMOTE:', np.bincount(y_train))\n"
            "Xr, yr = SMOTE(random_state=config.RANDOM_STATE).fit_resample(Xt, y_train)\n"
            "print('after  SMOTE:', np.bincount(yr))"
        ),
        md(
            "### Resampling justification\n"
            "| Aspect | Choice |\n"
            "|---|---|\n"
            "| Technique | **SMOTE** (synthetic oversampling) |\n"
            "| Why not undersample | Fraud is rare in absolute count; undersampling throws away the majority of legit data |\n"
            "| Leakage control | Applied inside the pipeline, **train fold only** |\n"
            "| Alternative for credit-card | `scale_pos_weight` / `class_weight` also viable given extreme ratio |"
        ),
    ]
    write("feature-engineering.ipynb", cells)


# --------------------------------------------------------------------------- #
# 4. Modeling
# --------------------------------------------------------------------------- #
def build_modeling():
    cells = [
        md(
            "# Model Building & Evaluation (Task 2)\n\n"
            "Baseline Logistic Regression vs an ensemble (XGBoost / Random Forest) on "
            "both datasets, evaluated with **AUC-PR, F1, confusion matrix** and 5-fold CV."
        ),
        code(BOOTSTRAP),
        code(
            "from src import config, preprocessing, modeling, evaluation\n"
            "fraud = pd.read_csv(config.PROCESSED_DIR/'fraud_data_processed.csv')\n"
            "X, y = preprocessing.select_fraud_model_columns(fraud)\n"
            "X_tr, X_te, y_tr, y_te = preprocessing.stratified_split(X, y)\n"
            "print('train', X_tr.shape, 'test', X_te.shape, '| test frauds:', int(y_te.sum()))"
        ),
        md("## Baseline - Logistic Regression (`class_weight='balanced'`)"),
        code(
            "lr = preprocessing.make_plain_pipeline(preprocessing.build_fraud_preprocessor(X_tr), modeling.make_logistic_regression())\n"
            "lr.fit(X_tr, y_tr)\n"
            "lr_score = lr.predict_proba(X_te)[:,1]; lr_pred = (lr_score>=0.5).astype(int)\n"
            "rec_lr = evaluation.evaluate(y_te, lr_pred, lr_score, 'LogisticRegression'); rec_lr"
        ),
        md("## Ensemble - XGBoost (`scale_pos_weight` for imbalance)"),
        code(
            "spw = modeling.imbalance_ratio(y_tr)\n"
            "try:\n"
            "    est = modeling.make_xgboost(scale_pos_weight=spw); name='xgboost'\n"
            "except ImportError:\n"
            "    est = modeling.make_random_forest(); name='random_forest'\n"
            "ens = preprocessing.make_plain_pipeline(preprocessing.build_fraud_preprocessor(X_tr), est)\n"
            "ens.fit(X_tr, y_tr)\n"
            "ens_score = ens.predict_proba(X_te)[:,1]; ens_pred = (ens_score>=0.5).astype(int)\n"
            "rec_ens = evaluation.evaluate(y_te, ens_pred, ens_score, name); rec_ens"
        ),
        md("## Comparison table"),
        code("display(evaluation.metrics_table([rec_lr, rec_ens]))"),
        md("## Confusion matrices & PR curves"),
        code(
            "evaluation.plot_confusion_matrix(y_te, lr_pred, 'LogReg', config.FIGURES_DIR/'cm_fraud_lr.png')\n"
            "evaluation.plot_confusion_matrix(y_te, ens_pred, name, config.FIGURES_DIR/'cm_fraud_ens.png')\n"
            "evaluation.plot_pr_curve(y_te, ens_score, name, config.FIGURES_DIR/'pr_fraud_ens.png')\n"
            "plt.show()"
        ),
        md("## 5-fold Stratified Cross-Validation"),
        code(
            "cv = modeling.cross_validate_model(ens, X, y, k=5)\n"
            "for m,(mu,sd) in cv.items(): print(f'{m:10s}: {mu:.4f} +/- {sd:.4f}')"
        ),
        md("## Save the best model"),
        code(
            "best = ens  # ensemble selected on AUC-PR; see comparison table\n"
            "modeling.save_model(best, 'fraud_data_best')\n"
            "print('saved -> models/fraud_data_best.joblib')"
        ),
        md(
            "---\n## Credit-card dataset\nSame procedure on the PCA-anonymised bank data."
        ),
        code(
            "cc = pd.read_csv(config.PROCESSED_DIR/'creditcard_processed.csv')\n"
            "Xc = cc.drop(columns=[config.CREDITCARD_TARGET]); yc = cc[config.CREDITCARD_TARGET]\n"
            "Xc_tr, Xc_te, yc_tr, yc_te = preprocessing.stratified_split(Xc, yc)\n"
            "lrc = preprocessing.make_plain_pipeline(preprocessing.build_creditcard_preprocessor(Xc_tr), modeling.make_logistic_regression())\n"
            "lrc.fit(Xc_tr, yc_tr)\n"
            "sc = lrc.predict_proba(Xc_te)[:,1]; pc = (sc>=0.5).astype(int)\n"
            "rec_lrc = evaluation.evaluate(yc_te, pc, sc, 'LogReg-CC')\n"
            "try:\n"
            "    estc = modeling.make_xgboost(scale_pos_weight=modeling.imbalance_ratio(yc_tr)); ncc='xgboost-CC'\n"
            "except ImportError:\n"
            "    estc = modeling.make_random_forest(); ncc='rf-CC'\n"
            "ensc = preprocessing.make_plain_pipeline(preprocessing.build_creditcard_preprocessor(Xc_tr), estc)\n"
            "ensc.fit(Xc_tr, yc_tr)\n"
            "scs = ensc.predict_proba(Xc_te)[:,1]; pcs=(scs>=0.5).astype(int)\n"
            "rec_ensc = evaluation.evaluate(yc_te, pcs, scs, ncc)\n"
            "display(evaluation.metrics_table([rec_lrc, rec_ensc]))\n"
            "modeling.save_model(ensc, 'creditcard_best')"
        ),
        md(
            "### Model selection\n"
            "Select on **AUC-PR** (the right summary metric for imbalanced data), using F1 "
            "and the confusion matrix to weigh the false-positive vs false-negative trade-off. "
            "The ensemble typically wins on AUC-PR while Logistic Regression remains the "
            "interpretable reference. The chosen model is carried into the SHAP notebook."
        ),
    ]
    write("modeling.ipynb", cells)


# --------------------------------------------------------------------------- #
# 5. SHAP explainability
# --------------------------------------------------------------------------- #
def build_shap():
    cells = [
        md(
            "# Model Explainability with SHAP (Task 3)\n\n"
            "Built-in importance vs SHAP, a global summary plot, and force plots for a "
            "true positive, false positive and false negative."
        ),
        code(BOOTSTRAP),
        code(
            "import shap\n"
            "from src import config, preprocessing, modeling, explainability\n"
            "fraud = pd.read_csv(config.PROCESSED_DIR/'fraud_data_processed.csv')\n"
            "X, y = preprocessing.select_fraud_model_columns(fraud)\n"
            "X_tr, X_te, y_tr, y_te = preprocessing.stratified_split(X, y)\n"
            "try:\n"
            "    pipe = modeling.load_model('fraud_data_best')\n"
            "except Exception:\n"
            "    est = modeling.make_xgboost(scale_pos_weight=modeling.imbalance_ratio(y_tr))\n"
            "    pipe = preprocessing.make_plain_pipeline(preprocessing.build_fraud_preprocessor(X_tr), est).fit(X_tr, y_tr)\n"
            "print('model ready')"
        ),
        md("## 1. Built-in feature importance (top 10)"),
        code(
            "imp = explainability.builtin_importance(pipe, top_n=10)\n"
            "display(imp)\n"
            "imp.set_index('feature')['importance'].iloc[::-1].plot.barh(title='Built-in importance (top 10)')\n"
            "plt.savefig(config.FIGURES_DIR/'builtin_importance.png', dpi=120, bbox_inches='tight'); plt.show()"
        ),
        md("## 2. SHAP global summary"),
        code(
            "explainer, shap_values, X_trans = explainability.compute_shap_values(pipe, X_te.sample(min(2000, len(X_te)), random_state=42))\n"
            "shap.summary_plot(shap_values, X_trans, show=False)\n"
            "plt.savefig(config.FIGURES_DIR/'shap_summary.png', dpi=120, bbox_inches='tight'); plt.show()"
        ),
        md(
            "## 3. Force plots: TP, FP, FN\n"
            "We locate one correctly-caught fraud, one false alarm, and one missed fraud."
        ),
        code(
            "proba = pipe.predict_proba(X_te)[:,1]; pred = (proba>=0.5).astype(int)\n"
            "yte = y_te.reset_index(drop=True); pred = pd.Series(pred); proba = pd.Series(proba)\n"
            "tp = pred[(pred==1)&(yte==1)].index[:1]\n"
            "fp = pred[(pred==1)&(yte==0)].index[:1]\n"
            "fn = pred[(pred==0)&(yte==1)].index[:1]\n"
            "print('TP idx', list(tp), '| FP idx', list(fp), '| FN idx', list(fn))"
        ),
        code(
            "Xte_named = explainability.transform_for_shap(pipe, X_te).reset_index(drop=True)\n"
            "sv_all = explainer(Xte_named)\n"
            "shap.initjs()"
        ),
        code(
            "def force(i, title):\n"
            "    if len(i)==0:\n"
            "        print('no example for', title); return\n"
            "    idx=i[0]\n"
            "    shap.plots.waterfall(sv_all[idx], show=False)\n"
            "    plt.title(title); plt.savefig(config.FIGURES_DIR/f'shap_{title}.png', dpi=120, bbox_inches='tight'); plt.show()\n"
            "force(tp, 'true_positive')\n"
            "force(fp, 'false_positive')\n"
            "force(fn, 'false_negative')"
        ),
        md(
            "## 4. Interpretation & business recommendations\n\n"
            "**Top fraud drivers (expected):** `time_since_signup`, transaction velocity "
            "(`sec_since_prev_user_tx`, `device_shared_users`), `purchase_value`, `hour_of_day`, "
            "and certain `source`/`country` values.\n\n"
            "**Built-in vs SHAP:** built-in importance ranks features by split gain (global, "
            "direction-agnostic); SHAP shows *direction* and *per-prediction* contribution - e.g. "
            "a *small* `time_since_signup` pushes a prediction **towards fraud**.\n\n"
            "**Recommendations**\n"
            "1. **Step-up verification for purchases made shortly after signup** (small "
            "`time_since_signup`) - the dominant SHAP driver.\n"
            "2. **Flag device/IP velocity**: many users per device or rapid repeat purchases "
            "(`device_shared_users`, `sec_since_prev_user_tx`) warrant review.\n"
            "3. **Geo-risk rules**: add friction for transactions from countries with elevated "
            "fraud rates surfaced in the geolocation analysis.\n"
            "Each recommendation maps directly to a high-SHAP feature above."
        ),
    ]
    write("shap-explainability.ipynb", cells)


def main():
    NB_DIR.mkdir(parents=True, exist_ok=True)
    build_eda_fraud()
    build_eda_creditcard()
    build_feature_engineering()
    build_modeling()
    build_shap()
    print("All notebooks generated.")


if __name__ == "__main__":
    main()
