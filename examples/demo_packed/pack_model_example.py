"""Как упаковать модель вместе с обработкой. Запускается как есть: python pack_model_example.py

В жизни первые два блока — это ваш ноутбук с обучением; упаковка — последний блок.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from scoring_kit import pack_model

# --- 1. Сырая выборка: то же, что вернёт SQL процесса (здесь — синтетика)
rng = np.random.default_rng(0)
n = 5000
raw_df = pd.DataFrame({
    "client_id": np.arange(n),
    "report_dt": "2026-09-01",
    "debt_amt": rng.gamma(2.0, 20000.0, n),
    "dpd_days": rng.integers(0, 120, n).astype(float),
    "product_cd": rng.choice(["CC", "CL", "MT"], n),
})
raw_df.loc[rng.random(n) < 0.1, "dpd_days"] = np.nan
target = (rng.random(n) < 1 / (1 + np.exp(3 - raw_df["dpd_days"].fillna(0) / 40))).astype(int)


# --- 2. Своя обработка и обучение
def prepare(df):
    """Сырые колонки -> фичи модели. Доступны np и pd; остальное импортируйте внутри функции."""
    df["debt_log"] = np.log1p(df["debt_amt"])
    df["dpd_days"] = df["dpd_days"].fillna(-1)
    return df


FEATURES = ["debt_log", "dpd_days", "product_cd"]      # в том порядке, в котором обучается модель
CAT_FEATURES = ["product_cd"]

X = prepare(raw_df.copy())[FEATURES]
X["product_cd"] = X["product_cd"].astype("category")
model = lgb.LGBMClassifier(n_estimators=50, verbose=-1).fit(X, target)

# --- 3. Упаковка: модель + обработка + типы фичей в одном файле
packed = pack_model(
    model,
    features=FEATURES,
    cat_features=CAT_FEATURES,
    prepare=prepare,
    sample=raw_df.head(1000),       # по нему определяются входные колонки и сверяются скоры
    path="model_packed.pkl",        # этот файл загрузить в Model Registry
)

# Упакованная модель принимает сырую выборку и возвращает скор — так её вызовет и процесс.
print(packed.predict(raw_df.head(3)))
