"""Сквозной прогон сгенерированного DAG'а и сверка со старым пайплайном."""

import joblib
import numpy as np
import pandas as pd
import pytest

from scoring_kit.debug import operator_predictor, run_debug
from scoring_kit.render import render_dir
from tests.conftest import CAT_FEATURES, DEMO, FEATURES
from tests.legacy import legacy_scores
from tests.test_spec import make_dir


@pytest.fixture()
def data_csv(tmp_path, score_frame):
    path = tmp_path / "sample.csv"
    score_frame.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("batch_size", [700, 1000, 100000])
def test_scores_match_legacy_pipeline(tmp_path, data_csv, model_path, batch_size):
    """Главный критерий приёмки: скоры совпадают со старым кодом бит в бит."""
    d = make_dir(tmp_path, model__batch_size=batch_size)
    result = run_debug(d, data_csv, [model_path], tmp_path / "out")

    model = joblib.load(model_path)
    expected = legacy_scores(model, data_csv, FEATURES, CAT_FEATURES, batch_size=1000)
    got = result.scored["score"]
    assert len(got) == len(expected) == 2500
    np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())


def test_output_prepared_for_write(tmp_path, data_csv, model_path, score_frame):
    result = run_debug(DEMO, data_csv, [model_path], tmp_path / "out")
    df = result.to_write["sandbox.demo_scores_fresh"]
    assert list(df.columns) == ["client_id", "account_id", "report_dt", "segment", "reason_cd", "refuse_flg", "score"]
    assert str(df["client_id"].dtype) == "Int64"
    assert df["client_id"].iloc[0] == score_frame["client_id"].iloc[0]
    assert str(df["reason_cd"].dtype) == "Int64" and df["reason_cd"].isna().sum() > 0
    assert df["score"].between(0, 1).all()
    assert "truncate table sandbox.demo_scores_fresh;" in result.sql["sandbox.demo_scores_fresh"]
    assert (tmp_path / "out" / "demo_scoring.py").exists()


def test_limit(tmp_path, data_csv, model_path):
    result = run_debug(DEMO, data_csv, [model_path], tmp_path / "out", limit=100)
    assert len(result.scored) == 100


def test_duplicate_key_stops_at_read(tmp_path, score_frame, model_path):
    bad = pd.concat([score_frame, score_frame.head(3)])
    path = tmp_path / "dups.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="не уникален"):
        run_debug(DEMO, path, [model_path], tmp_path / "out")


def test_missing_feature_stops_at_read(tmp_path, score_frame, model_path):
    path = tmp_path / "nofeat.csv"
    score_frame.drop(columns=["debt_sum"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="debt_sum"):
        run_debug(DEMO, path, [model_path], tmp_path / "out")


def test_predictor_as_operator_sees_it():
    """BatchInferenceOperator вырезает "BasePredictor" и переименовывает класс; в predict.py нет имён модуля."""
    _, code = render_dir(DEMO)
    cls = operator_predictor(code)
    assert cls.__name__ == "Predictor"
    assert cls.__bases__ == (object,)
    assert cls.features[0] == "segment"


def test_sink_column_missing_in_source_stops_at_read(tmp_path, score_frame, model_path):
    """Колонка приёмника, которой нет в выборке, ловится на чтении, а не после скоринга."""
    path = tmp_path / "nodate.csv"
    score_frame.drop(columns=["report_dt"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="report_dt"):
        run_debug(DEMO, path, [model_path], tmp_path / "out")


def test_output_columns_are_not_required_in_source(tmp_path, data_csv, model_path, score_frame):
    d = make_dir(tmp_path, model__output_columns=["report_dt"])
    path = tmp_path / "nodate2.csv"
    score_frame.drop(columns=["report_dt"]).to_csv(path, index=False)
    # предиктор report_dt не создаёт -> упадёт уже на записи, но чтение проходит
    with pytest.raises(ValueError, match="для записи нет колонок"):
        run_debug(d, path, [model_path], tmp_path / "out")
