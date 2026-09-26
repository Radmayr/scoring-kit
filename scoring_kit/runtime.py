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
