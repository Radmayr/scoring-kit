"""Функции, которые исполняются внутри подов Airflow.

Генератор вставляет их исходный код (inspect.getsource) внутрь колбэков и метода
predict, потому что в под передаётся только тело функции/класса. Поэтому каждая
функция здесь самодостаточна: все импорты внутри, никаких ссылок на имена модуля.

Переменная окружения SCORING_KIT_WORK_ROOT нужна только локальной отладке —
она подменяет корень путей /work/...; в Airflow она не задана.
"""


def sk_read_frame(path):
    import os
    from pathlib import Path

    import pandas as pd

    path = Path(os.environ.get("SCORING_KIT_WORK_ROOT", "") + str(path))
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def sk_write_frame(df, path):
    import os
    from pathlib import Path

    path = Path(os.environ.get("SCORING_KIT_WORK_ROOT", "") + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"[scoring-kit] записано {len(df)} строк в {path}")


def sk_normalize_decimals(df):
    """numeric из Greenplum приходит как object с Decimal — переводим в float64."""
    import decimal

    import pandas as pd

    for c in df.columns:
        s = df[c]
        if s.dtype == object:
            non_null = s.dropna()
            if len(non_null) and isinstance(non_null.iloc[0], decimal.Decimal):
                df[c] = pd.to_numeric(s, errors="raise").astype("float64")
    return df


def sk_check_source(df, min_rows, required_columns, unique_key):
    import pandas as pd

    print(f"[scoring-kit] входная выборка: {df.shape[0]} строк x {df.shape[1]} колонок")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"[scoring-kit] в выборке нет колонок {missing}. Есть: {list(df.columns)}"
        )
    if len(df) < min_rows:
        raise ValueError(f"[scoring-kit] в выборке {len(df)} строк, ожидалось не меньше {min_rows}")
    if unique_key:
        dup_mask = df.duplicated(subset=list(unique_key), keep=False)
        if dup_mask.any():
            sample = df.loc[dup_mask, list(unique_key)].head(10)
            raise ValueError(
                f"[scoring-kit] ключ {list(unique_key)} не уникален: {int(dup_mask.sum())} строк "
                f"с дублями, примеры:\n{sample.to_string(index=False)}"
            )
    if required_columns:
        nulls = df[list(required_columns)].isna().mean()
        nulls = nulls[nulls > 0].sort_values(ascending=False)
        if len(nulls):
            with pd.option_context("display.max_rows", None):
                print(f"[scoring-kit] доля пропусков в фичах:\n{nulls.round(4).to_string()}")


def sk_prepare_features(df, features, cat_features, num_dtype, cat_as):
    """Приводит фичи к типам контракта; остальные колонки не трогает.

    Нужна потому, что при передаче через csv типы восстанавливаются заново в каждом
    файле/батче: int-колонка с пропуском становится float, категория — числом и т.д.
    """
    import pandas as pd
    from pandas.api.types import (
        is_bool_dtype,
        is_datetime64_any_dtype,
        is_numeric_dtype,
        is_timedelta64_dtype,
    )

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"[scoring-kit] нет фичей {missing}. Есть: {list(df.columns)}")

    out = df.copy()
    cats = set(cat_features)
    for c in features:
        s = out[c]
        if s.dtype == object:
            # numeric/decimal из Greenplum и Spark приходит объектами Decimal
            import decimal

            non_null = s.dropna()
            if len(non_null) and isinstance(non_null.iloc[0], decimal.Decimal):
                s = pd.to_numeric(s, errors="coerce").astype("float64")
        if c in cats:
            if is_bool_dtype(s):
                s = s.astype("Int64")
            elif is_numeric_dtype(s):
                non_null = s.dropna()
                if len(non_null) and (non_null == non_null.round()).all():
                    # 1.0 и 1 должны быть одной категорией независимо от пропусков в батче.
                    s = s.astype("Int64")
            if cat_as == "category":
                out[c] = s.astype("category")
            else:
                out[c] = s.astype(object).where(s.notna(), "nan").astype(str)
        else:
            if is_datetime64_any_dtype(s) or is_timedelta64_dtype(s):
                raise ValueError(
                    f"[scoring-kit] фича {c} имеет тип {s.dtype}; даты переводите в числа в SQL"
                )
            if is_bool_dtype(s):
                s = s.astype("float64")
            converted = pd.to_numeric(s, errors="coerce")
            broken = s.notna() & converted.isna()
            if broken.any():
                raise ValueError(
                    f"[scoring-kit] числовая фича {c}: {int(broken.sum())} значений не число, "
                    f"например {list(s[broken].astype(str).unique()[:5])}"
                )
            out[c] = converted.astype(num_dtype)
    return out


def sk_check_scores(result, n_rows, score_column, score_range):
    import pandas as pd

    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"[scoring-kit] predict должен вернуть DataFrame, а вернул {type(result)}")
    if len(result) != n_rows:
        raise ValueError(f"[scoring-kit] predict вернул {len(result)} строк вместо {n_rows}")
    if score_column not in result.columns:
        raise ValueError(f"[scoring-kit] в результате predict нет колонки '{score_column}'")
    s = pd.to_numeric(result[score_column], errors="coerce")
    if s.isna().any():
        raise ValueError(f"[scoring-kit] в '{score_column}' {int(s.isna().sum())} пустых/нечисловых значений")
    if score_range is not None:
        lo, hi = score_range
        out = (s < lo) | (s > hi)
        if out.any():
            raise ValueError(
                f"[scoring-kit] {int(out.sum())} значений '{score_column}' вне [{lo}, {hi}], "
                f"min={s.min()}, max={s.max()}"
            )


def sk_conform_to_columns(df, columns_types):
    """Оставляет колонки приёмника в объявленном порядке и сверяет их типы."""
    import pandas as pd

    missing = [c for c in columns_types if c not in df.columns]
    if missing:
        raise ValueError(
            f"[scoring-kit] для записи нет колонок {missing}. Есть: {list(df.columns)}"
        )
    out = df[list(columns_types)].copy()
    int_types = {"int", "integer", "bigint", "smallint", "int2", "int4", "int8"}
    float_types = {"numeric", "decimal", "float", "float4", "float8", "double precision", "real"}
    for c, sql_type in columns_types.items():
        base = sql_type.lower().split("(")[0].strip()
        s = out[c]
        if base in int_types:
            num = pd.to_numeric(s, errors="coerce")
            broken = s.notna() & num.isna()
            fractional = num.notna() & (num != num.round())
            if broken.any() or fractional.any():
                bad = s[broken | fractional].astype(str).unique()[:5]
                raise ValueError(f"[scoring-kit] колонка {c} ({sql_type}): не целые значения {list(bad)}")
            out[c] = num.astype("Int64")
        elif base in float_types:
            num = pd.to_numeric(s, errors="coerce")
            broken = s.notna() & num.isna()
            if broken.any():
                raise ValueError(
                    f"[scoring-kit] колонка {c} ({sql_type}): не числа {list(s[broken].astype(str).unique()[:5])}"
                )
            out[c] = num.astype("float64")
        elif base in {"date", "timestamp", "timestamptz"} or base.startswith("timestamp"):
            parsed = pd.to_datetime(s, errors="coerce")
            broken = s.notna() & parsed.isna()
            if broken.any():
                raise ValueError(
                    f"[scoring-kit] колонка {c} ({sql_type}): не даты {list(s[broken].astype(str).unique()[:5])}"
                )
            if base == "date":
                # datetime.date -> date32 в stg: колонка получится date, а вставка в varchar даст '2026-09-01'.
                out[c] = pd.Series(
                    [d.date() if pd.notna(d) else None for d in parsed], index=out.index, dtype=object
                )
            else:
                out[c] = parsed
    print(f"[scoring-kit] к записи {len(out)} строк, колонки: {list(out.columns)}")
    return out


# ---------------------------------------------------------------- встроенный предиктор


def sk_load_model(path):
    """Модель из артефакта реестра: joblib/pickle; нативные форматы бустингов — по расширению."""
    import pickle

    path = str(path)
    if path.endswith((".txt", ".lgb")):
        import lightgbm as lgb

        return lgb.Booster(model_file=path)
    if path.endswith(".cbm"):
        import catboost

        model = catboost.CatBoost()
        model.load_model(path)
        return model
    try:
        import joblib

        return joblib.load(path)
    except ImportError:
        with open(path, "rb") as f:
            return pickle.load(f)


def sk_model_scores(model, X, output_kind):
    """proba — вероятность класса 1; predict — сырой predict (регрессия, метки)."""
    import numpy as np

    if output_kind == "proba":
        if hasattr(model, "predict_proba"):
            values = np.asarray(model.predict_proba(X))
            return values[:, 1] if values.ndim == 2 else values
        # lightgbm.Booster бинарной модели сразу отдаёт вероятность
        return np.asarray(model.predict(X))
    return np.asarray(model.predict(X))


# ---------------------------------------------------------------- multi_model: слияние job'ов


def sk_merge_outputs(paths, key, score_columns):
    """Первый выход берётся целиком, из остальных — только их колонки скоров по ключу."""
    import os
    from pathlib import Path

    import pandas as pd

    root = os.environ.get("SCORING_KIT_WORK_ROOT", "")
    frames = [pd.read_csv(Path(root + p), low_memory=False) for p in paths]
    base = frames[0]
    n = len(base)
    for frame, cols in zip(frames[1:], score_columns[1:]):
        if len(frame) != n:
            raise ValueError(f"[scoring-kit] job'ы вернули разное число строк: {n} и {len(frame)}")
        clash = [c for c in cols if c in base.columns]
        if clash:
            raise ValueError(f"[scoring-kit] колонки {clash} уже есть в результате первого job'а")
        base = base.merge(frame[list(key) + list(cols)], on=list(key), how="left", validate="one_to_one")
    if len(base) != n:
        raise ValueError(f"[scoring-kit] после слияния {len(base)} строк вместо {n}: проверьте ключ {list(key)}")
    print(f"[scoring-kit] слито {len(frames)} результатов, {n} строк")
    return base


# ---------------------------------------------------------------- fit_apply: калибровка


def sk_calibrate(dev, apply, cfg):
    """Калибровка по сегментам и когортам.

    Для каждого сегмента (декартово произведение значений полей; список значений = группа)
    когорты dev сортируются; для когорты на позиции i калибратор обучается на когортах с
    позициями i + train_offsets и применяется к строкам apply той же когорты.
    Возвращает (apply + колонка калиброванного скора, отчёт по ячейкам).
    """
    import itertools

    import numpy as np
    import pandas as pd

    score, target, output = cfg["score"], cfg["target"], cfg["output"]
    cohort, offsets, method = cfg["cohort_column"], sorted(cfg["train_offsets"]), cfg["method"]
    first = -min(offsets)
    fields = list(cfg["segments"])

    def fit(x, y):
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(x, y)
            return model.predict
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        model.fit(x.reshape(-1, 1), y)
        return lambda v: model.predict_proba(np.asarray(v, dtype=float).reshape(-1, 1))[:, 1]

    def brier(y, p):
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        return float(np.mean((p - y) ** 2)) if len(y) else None

    def mask(frame, field, value):
        if isinstance(value, (list, tuple)):
            return frame[field].isin(list(value)).to_numpy()
        return (frame[field] == value).to_numpy()

    calibrated = pd.Series(np.nan, index=apply.index, dtype="float64")
    report = []
    for combo in itertools.product(*(cfg["segments"][f] for f in fields)):
        segment = " & ".join(f"{f}={v}" for f, v in zip(fields, combo))
        dm = np.ones(len(dev), dtype=bool)
        am = np.ones(len(apply), dtype=bool)
        for f, v in zip(fields, combo):
            dm &= mask(dev, f, v)
            am &= mask(apply, f, v)
        dev_seg, apply_seg = dev[dm], apply[am]
        cohorts = sorted(dev_seg[cohort].dropna().unique())
        print(f"[scoring-kit] сегмент {segment}: dev {len(dev_seg)} строк, когорт {len(cohorts)}, apply {len(apply_seg)}")
        if len(cohorts) <= first:
            report.append({
                "segment": segment, "cohort": None, "status": "мало когорт в dev", "train_cohorts": None,
                "train_rows": 0, "test_rows": 0, "apply_rows": int(len(apply_seg)),
                "brier_raw": None, "brier_calibrated": None,
            })
            continue
        for i in range(first, len(cohorts)):
            test_cohort = cohorts[i]
            train_cohorts = [cohorts[i + o] for o in offsets]
            train = dev_seg[dev_seg[cohort].isin(train_cohorts)]
            test = dev_seg[dev_seg[cohort] == test_cohort]
            target_rows = apply_seg[apply_seg[cohort] == test_cohort]
            row = {
                "segment": segment, "cohort": str(test_cohort), "train_cohorts": ",".join(map(str, train_cohorts)),
                "train_rows": int(len(train)), "test_rows": int(len(test)), "apply_rows": int(len(target_rows)),
                "brier_raw": None, "brier_calibrated": None,
            }
            if len(train) == 0 or train[target].nunique(dropna=True) < 2:
                row["status"] = "нет данных для обучения"
                report.append(row)
                continue
            predict = fit(train[score].astype(float).to_numpy(), train[target].astype(float).to_numpy())
            if len(test):
                row["brier_raw"] = brier(test[target], test[score])
                row["brier_calibrated"] = brier(test[target], predict(test[score].astype(float).to_numpy()))
            if len(target_rows):
                calibrated.loc[target_rows.index] = predict(target_rows[score].astype(float).to_numpy())
            row["status"] = "ok"
            report.append(row)

    result = apply.copy()
    if cfg["uncalibrated"] == "copy_score":
        calibrated = calibrated.fillna(apply[score].astype(float))
    result[output] = calibrated
    covered = float(calibrated.notna().mean()) if len(calibrated) else 1.0
    print(f"[scoring-kit] откалибровано {covered:.2%} строк apply")
    return result, pd.DataFrame(report)


# ---------------------------------------------------------------- engine: dlh


def sk_dlh_check_source(df, min_rows):
    """df — одна строка: n_rows, dup_keys (результат SQL-проверки в Trino)."""
    row = df.iloc[0]
    n_rows, dup_keys = int(row["n_rows"]), int(row["dup_keys"])
    print(f"[scoring-kit] выборка: {n_rows} строк, повторяющихся значений ключа: {dup_keys}")
    if n_rows < min_rows:
        raise ValueError(f"[scoring-kit] в выборке {n_rows} строк, ожидалось не меньше {min_rows}")
    if dup_keys:
        raise ValueError(f"[scoring-kit] ключ не уникален: {dup_keys} значений ключа повторяются")


def sk_dlh_check_target(df, score_range):
    """df — одна строка: n_rows, n_source, n_score, min_score, max_score."""
    row = df.iloc[0]
    n_rows, n_source, n_score = int(row["n_rows"]), int(row["n_source"]), int(row["n_score"])
    print(
        f"[scoring-kit] результат: {n_rows} строк (на входе {n_source}), скоров {n_score}, "
        f"min={row['min_score']}, max={row['max_score']}"
    )
    if n_rows != n_source:
        raise ValueError(f"[scoring-kit] в результате {n_rows} строк, а на входе было {n_source}")
    if n_score != n_rows:
        raise ValueError(f"[scoring-kit] {n_rows - n_score} пустых скоров")
    if score_range is not None and n_rows:
        lo, hi = score_range
        if float(row["min_score"]) < lo or float(row["max_score"]) > hi:
            raise ValueError(f"[scoring-kit] скор вне [{lo}, {hi}]: min={row['min_score']}, max={row['max_score']}")
