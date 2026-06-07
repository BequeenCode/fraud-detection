"""SHAP-based explainability helpers for the best (tree-ensemble) model.

Provides utilities to (1) read built-in feature importances out of a fitted
pipeline, and (2) compute SHAP values on the *transformed* feature space with
correct feature names recovered from the ColumnTransformer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_feature_names(preprocessor) -> list[str]:
    """Recover output feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - very old sklearn
        return [f"f{i}" for i in range(preprocessor.transform_shape_[1])]


def builtin_importance(pipeline, top_n: int = 10) -> pd.DataFrame:
    """Return the top-N built-in feature importances from a fitted pipeline.

    Works for tree ensembles (``feature_importances_``) and linear models
    (absolute coefficients).
    """
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = get_feature_names(pre)

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).ravel()
    else:  # pragma: no cover
        raise AttributeError("Model exposes neither feature_importances_ nor coef_.")

    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def transform_for_shap(pipeline, X) -> pd.DataFrame:
    """Apply the pipeline's preprocessing and return a named DataFrame for SHAP."""
    pre = pipeline.named_steps["preprocess"]
    transformed = pre.transform(X)
    names = get_feature_names(pre)
    return pd.DataFrame(transformed, columns=names, index=getattr(X, "index", None))


def compute_shap_values(pipeline, X_sample):
    """Compute SHAP values for the model on a preprocessed sample.

    Returns ``(explainer, shap_values, X_transformed)``. Uses ``shap.Explainer``
    which auto-selects TreeExplainer for gradient-boosted / forest models.
    """
    import shap

    model = pipeline.named_steps["model"]
    X_trans = transform_for_shap(pipeline, X_sample)
    explainer = shap.Explainer(model, X_trans)
    shap_values = explainer(X_trans)
    return explainer, shap_values, X_trans
