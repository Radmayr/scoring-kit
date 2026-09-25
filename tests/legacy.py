"""Код предобработки из старого пайплайна (Bankrupt_phis) — эталон для сверки скоров."""

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)


def make_lgbm_ready(X: pd.DataFrame, cat_cols=None):
    X2 = X.copy()
    if cat_cols is None:
        cat_cols = X2.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c in X2.columns]
    for c in cat_cols:
        X2[c] = X2[c].astype("category")
    for c in X2.columns:
        if c in cat_cols:
            continue
        s = X2[c]
        if is_datetime64_any_dtype(s) or is_timedelta64_dtype(s):
            X2[c] = s.view("int64").astype("float32")
        elif is_bool_dtype(s):
            X2[c] = s.astype("int8")
        elif is_numeric_dtype(s):
            X2[c] = s.astype("float32")
        else:
            X2[c] = pd.to_numeric(s, errors="coerce").astype("float32")
    return X2, cat_cols


def legacy_scores(model, csv_path, features, cat_features, batch_size) -> pd.Series:
    """Как старый пайплайн: csv -> батчи -> make_lgbm_ready -> predict_proba."""
    num = [f for f in features if f not in cat_features]
    parts = []
    for batch in pd.read_csv(csv_path, chunksize=batch_size, low_memory=False):
        X, _ = make_lgbm_ready(batch[cat_features + num], cat_cols=cat_features)
        parts.append(pd.Series(model.predict_proba(X[features])[:, 1]))
    return pd.concat(parts, ignore_index=True)
