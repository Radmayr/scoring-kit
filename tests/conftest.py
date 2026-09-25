from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from tests.legacy import make_lgbm_ready

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo_scoring"

FEATURES = ["segment", "bal_avg_3m", "dpd_sum_12m", "reason_cd", "debt_sum", "refuse_flg", "req_cnt_3m", "open_accnt_cnt"]
CAT_FEATURES = ["segment", "reason_cd", "refuse_flg"]


def make_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "client_id": np.arange(n) + 10_000_000_000,  # больше int4 — проверка bigint
            "account_id": np.arange(n) + 1,
            "report_dt": pd.Timestamp("2026-09-01").strftime("%Y-%m-%d"),
            "segment": rng.choice(["A", "B", "C"], n),
            "bal_avg_3m": rng.gamma(2.0, 5000.0, n),
            "dpd_sum_12m": rng.integers(0, 400, n).astype(float),
            "reason_cd": rng.choice([0, 1, 2], n).astype(float),
            "debt_sum": rng.gamma(1.5, 10000.0, n),
            "refuse_flg": rng.choice([0, 1], n),
            "req_cnt_3m": rng.integers(0, 10, n),
            "open_accnt_cnt": rng.integers(0, 5, n),
        }
    )
    # Пропуски сосредоточены в начале, чтобы батчи различались по составу типов.
    df.loc[: n // 10, "reason_cd"] = np.nan
    df.loc[rng.random(n) < 0.05, "bal_avg_3m"] = np.nan
    df.loc[rng.random(n) < 0.03, "segment"] = None
    logit = -3 + 0.002 * df["dpd_sum_12m"] + (df["segment"] == "C") * 1.0 + df["refuse_flg"] * 0.8
    df["target_6m"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture(scope="session")
def train_frame() -> pd.DataFrame:
    return make_frame(4000, seed=1)


@pytest.fixture(scope="session")
def score_frame() -> pd.DataFrame:
    return make_frame(2500, seed=2)


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, train_frame) -> Path:
    """Модель, обученная так же, как в старых ноутбуках: make_lgbm_ready + LGBMClassifier."""
    import lightgbm as lgb

    X, _ = make_lgbm_ready(train_frame[FEATURES], cat_cols=CAT_FEATURES)
    model = lgb.LGBMClassifier(n_estimators=60, num_leaves=15, random_state=0)
    model.fit(X[FEATURES], train_frame["target_6m"])
    path = tmp_path_factory.mktemp("model") / "model.pkl"
    joblib.dump(model, path)
    return path
