import datetime as dt
import decimal

import numpy as np
import pandas as pd
import pytest

from scoring_kit import runtime as rt


def test_check_source_ok(capsys):
    df = pd.DataFrame({"id": [1, 2], "f": [1.0, None]})
    rt.sk_check_source(df, 1, ["f"], ["id"])
    assert "доля пропусков" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"min_rows": 5, "required_columns": [], "unique_key": []}, "не меньше 5"),
        ({"min_rows": 1, "required_columns": ["nope"], "unique_key": []}, "нет колонок"),
        ({"min_rows": 1, "required_columns": [], "unique_key": ["id"]}, "не уникален"),
    ],
)
def test_check_source_errors(kwargs, message):
    df = pd.DataFrame({"id": [1, 1], "f": [1.0, 2.0]})
    with pytest.raises(ValueError, match=message):
        rt.sk_check_source(df, **kwargs)


def test_empty_frame_fails_min_rows():
    with pytest.raises(ValueError, match="0 строк"):
        rt.sk_check_source(pd.DataFrame({"f": []}), 1, ["f"], [])


def test_normalize_decimals():
    df = pd.DataFrame({"d": [decimal.Decimal("1.5"), None], "s": ["a", "b"]})
    out = rt.sk_normalize_decimals(df)
    assert out["d"].dtype == "float64"
    assert out["s"].dtype == object


def test_categories_consistent_across_batches():
    """Батч без пропусков (int) и с пропусками (float) дают одинаковые категории."""
    b1 = pd.DataFrame({"c": [0, 1, 2]})
    b2 = pd.DataFrame({"c": [0.0, np.nan, 2.0]})
    p1 = rt.sk_prepare_features(b1, ["c"], ["c"], "float32", "category")
    p2 = rt.sk_prepare_features(b2, ["c"], ["c"], "float32", "category")
    assert list(p1["c"].cat.categories) == [0, 1, 2]
    assert list(p2["c"].cat.categories) == [0, 2]
    assert p1["c"].cat.categories.dtype == p2["c"].cat.categories.dtype


def test_prepare_features_str_mode():
    df = pd.DataFrame({"c": [1.0, np.nan, 2.0], "s": ["x", None, "y"]})
    out = rt.sk_prepare_features(df, ["c", "s"], ["c", "s"], "float64", "str")
    assert list(out["c"]) == ["1", "nan", "2"]
    assert list(out["s"]) == ["x", "nan", "y"]


def test_prepare_features_numeric():
    df = pd.DataFrame({"n": ["1.5", None, "3"], "b": [True, False, True], "other": ["keep", "me", "!"]})
    out = rt.sk_prepare_features(df, ["n", "b"], [], "float32", "category")
    assert out["n"].dtype == "float32"
    assert out["b"].dtype == "float32"
    assert list(out["other"]) == ["keep", "me", "!"]
    assert df["n"].dtype == object  # вход не изменён


def test_prepare_features_rejects_garbage():
    with pytest.raises(ValueError, match="не число"):
        rt.sk_prepare_features(pd.DataFrame({"n": ["1", "abc"]}), ["n"], [], "float64", "category")
    with pytest.raises(ValueError, match="нет фичей"):
        rt.sk_prepare_features(pd.DataFrame({"n": [1]}), ["m"], [], "float64", "category")
    with pytest.raises(ValueError, match="даты"):
        rt.sk_prepare_features(pd.DataFrame({"d": pd.to_datetime(["2026-01-01"])}), ["d"], [], "float64", "category")


@pytest.mark.parametrize(
    "result, message",
    [
        ([1, 2], "DataFrame"),
        (pd.DataFrame({"score": [0.1]}), "строк вместо"),
        (pd.DataFrame({"x": [0.1, 0.2]}), "нет колонки"),
        (pd.DataFrame({"score": [0.1, None]}), "пустых"),
        (pd.DataFrame({"score": [0.1, 1.2]}), "вне"),
    ],
)
def test_check_scores_errors(result, message):
    with pytest.raises((ValueError, TypeError), match=message):
        rt.sk_check_scores(result, 2, "score", (0, 1))


def test_check_scores_no_range():
    rt.sk_check_scores(pd.DataFrame({"score": [-5.0, 7.0]}), 2, "score", None)


def test_conform_to_columns():
    df = pd.DataFrame(
        {
            "extra": [1, 2],
            "id": [10_000_000_000.0, np.nan],
            "dt": ["2026-09-01", "2026-09-02"],
            "s": ["a", None],
            "score": ["0.5", 0.25],
        }
    )
    out = rt.sk_conform_to_columns(df, {"id": "bigint", "dt": "date", "s": "varchar", "score": "numeric"})
    assert list(out.columns) == ["id", "dt", "s", "score"]
    assert str(out["id"].dtype) == "Int64" and out["id"][0] == 10_000_000_000
    assert out["dt"].tolist() == [dt.date(2026, 9, 1), dt.date(2026, 9, 2)]
    assert out["score"].tolist() == [0.5, 0.25]


def test_conform_errors():
    with pytest.raises(ValueError, match="не целые"):
        rt.sk_conform_to_columns(pd.DataFrame({"i": [1.5]}), {"i": "int"})
    with pytest.raises(ValueError, match="нет колонок"):
        rt.sk_conform_to_columns(pd.DataFrame({"i": [1]}), {"j": "int"})
    with pytest.raises(ValueError, match="не даты"):
        rt.sk_conform_to_columns(pd.DataFrame({"d": ["вчера"]}), {"d": "date"})


def test_frame_io_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SCORING_KIT_WORK_ROOT", str(tmp_path))
    df = pd.DataFrame({"a": [1, 2]})
    for name in ("data.csv", "data.parquet"):
        rt.sk_write_frame(df, f"/work/output/{name}")
        assert (tmp_path / "work" / "output" / name).exists()
        pd.testing.assert_frame_equal(rt.sk_read_frame(f"/work/output/{name}"), df)
