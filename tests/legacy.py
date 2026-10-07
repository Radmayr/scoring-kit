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


def legacy_multi_scores(models, csv_path, cat_features) -> pd.DataFrame:
    """Как процесс с 4 моделями: весь csv целиком (batch_size=None), make_lgbm_ready, predict_proba."""
    df = pd.read_csv(csv_path)
    for score_name, (model, features) in models.items():
        df_prep, _ = make_lgbm_ready(df, cat_cols=cat_features)
        df[score_name] = model.predict_proba(df_prep[features])[:, 1]
    return df


def legacy_calibration(dev_df: pd.DataFrame, apply_df: pd.DataFrame, segments: dict, score: str,
                       target: str, cohort: str, key: list, tmp_csv) -> pd.DataFrame:
    """Дословная логика калибровки из текущего процесса (transform_0 + inference_0.predict)."""
    import itertools

    from sklearn.isotonic import IsotonicRegression

    dev_df = dev_df.copy()
    apply_df = apply_df.copy()
    dev_df["sample"] = "dev"
    apply_df["sample"] = "apply"
    pd.concat([dev_df, apply_df]).to_csv(tmp_csv, index=False)
    df = pd.read_csv(tmp_csv)  # BatchInferenceOperator с batch_size=None читает csv целиком

    def calibrate_scores(frame, score_col, target_col):
        scores = frame[score_col].astype("float").values
        targets = frame[target_col].astype("float").values
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(scores, targets)
        return calibrator.predict(scores), calibrator

    def make_condition(x):
        if isinstance(x, str):
            return f"== '{x}'"
        if isinstance(x, (tuple, list)):
            return f"in {x}"
        return f"== {x}"

    df_dev = df[df["sample"] == "dev"].copy()
    df_apply = df[df["sample"] == "apply"].copy()
    df_calib = pd.DataFrame()
    keys = list(segments.keys())
    for segment in itertools.product(*segments.values()):
        condition = " AND \n".join([f"{k} {make_condition(v)}" for k, v in zip(keys, segment)])
        df_dev_segment = df_dev.query(condition).copy()
        df_apply_segment = df_apply.query(condition).copy()
        mths = sorted(df_dev_segment[cohort].unique())
        for i in range(3, len(mths)):
            test_month = mths[i]
            train_month = mths[i - 3 : i - 1]
            df_train = df_dev_segment[df_dev_segment[cohort].isin(train_month)].copy()
            _, isotonic_model = calibrate_scores(df_train, score, target)
            df_test2 = df_apply_segment[df_apply_segment[cohort] == test_month].copy()
            if df_test2.shape[0] > 0:
                df_test2[f"{score}_calib"] = isotonic_model.predict(df_test2[score])
                df_calib = pd.concat([df_calib, df_test2[[*key, f"{score}_calib"]]])
    return df_apply.merge(df_calib, on=key, how="left")


def legacy_scores(model, csv_path, features, cat_features, batch_size) -> pd.Series:
    """Как старый пайплайн: csv -> батчи -> make_lgbm_ready -> predict_proba."""
    num = [f for f in features if f not in cat_features]
    parts = []
    for batch in pd.read_csv(csv_path, chunksize=batch_size, low_memory=False):
        X, _ = make_lgbm_ready(batch[cat_features + num], cat_cols=cat_features)
        parts.append(pd.Series(model.predict_proba(X[features])[:, 1]))
    return pd.concat(parts, ignore_index=True)
