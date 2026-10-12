"""Упакованные модели: pack_model, загрузка без scoring-kit, паритет в Greenplum, DLH и multi_model."""

import pickle
import shutil
import subprocess
import sys
import textwrap

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import yaml

from scoring_kit import PackError, load_packed, pack_model
from scoring_kit.cli import main
from scoring_kit.debug import run_debug
from scoring_kit.spec import PipelineError, load_pipeline
from tests.conftest import CAT_FEATURES, DEMO, FEATURES, ROOT, make_frame
from tests.legacy import make_lgbm_ready

MULTI = ROOT / "examples" / "demo_multi"
DLH = ROOT / "examples" / "demo_scoring_dlh"

MODEL_FEATURES = ["segment", "dpd_sum_12m", "debt_log", "bal_filled", "refuse_flg"]
MODEL_CATS = ["segment", "refuse_flg"]
RAW_COLUMNS = ["segment", "bal_avg_3m", "dpd_sum_12m", "debt_sum", "refuse_flg"]


def prepare(df):
    df["debt_log"] = np.log1p(df["debt_sum"])
    df["bal_filled"] = df["bal_avg_3m"].fillna(-1.0)
    return df


def train(features=MODEL_FEATURES, cats=MODEL_CATS, seed=0, func=prepare):
    frame = make_frame(3000, seed=41 + seed)
    X, _ = make_lgbm_ready(func(frame.copy())[features], cat_cols=cats)
    model = lgb.LGBMClassifier(n_estimators=25 + 5 * seed, num_leaves=7, random_state=seed, verbose=-1)
    model.fit(X[features], frame["target_6m"])
    return model


def expected_scores(model, data, features=MODEL_FEATURES, cats=MODEL_CATS, func=prepare):
    """Как посчитал бы автор модели в ноутбуке: своя обработка, make_lgbm_ready, predict_proba."""
    X, _ = make_lgbm_ready(func(data.copy())[features], cat_cols=cats)
    return model.predict_proba(X[features])[:, 1]


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    model = train()
    path = tmp_path_factory.mktemp("packed") / "model_packed.pkl"
    obj = pack_model(model, MODEL_FEATURES, MODEL_CATS, prepare, sample=make_frame(400, seed=42),
                     num_dtype="float32", path=path, verbose=False)
    return model, obj, path


def write_pipeline(tmp_path, src, edit):
    d = tmp_path / src.name
    shutil.copytree(src, d)
    for extra in d.glob("predictor.py"):
        extra.unlink()
    raw = yaml.safe_load((d / "pipeline.yaml").read_text(encoding="utf-8"))
    edit(raw)
    (d / "pipeline.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return d


def as_packed(model: dict) -> dict:
    for key in ("features", "cat_features", "num_dtype", "cat_as", "output_kind"):
        model.pop(key, None)
    model["packed"] = True
    return model


# ---------------------------------------------------------------- сама упаковка


def test_pack_infers_input_columns_and_matches(packed):
    model, obj, path = packed
    assert obj.features == RAW_COLUMNS               # лишние колонки выборки в упаковку не попали
    data = make_frame(1500, seed=43)
    got = load_packed(path).predict(data)["score"].to_numpy()
    np.testing.assert_array_equal(got, expected_scores(model, data))
    text = obj.describe()
    assert "np.log1p" in text and "debt_log" in text and "float32" in text


def test_pack_without_prepare_is_plain_contract():
    features, cats = ["segment", "dpd_sum_12m", "debt_sum"], ["segment"]
    model = train(features, cats, func=lambda d: d)
    obj = pack_model(model, features, cats, num_dtype="float32", sample=make_frame(200, seed=44), path=None, verbose=False)
    assert obj.features == features
    data = make_frame(700, seed=45)
    np.testing.assert_array_equal(
        obj.predict(data)["score"].to_numpy(), expected_scores(model, data, features, cats, func=lambda d: d))


def test_pack_prepare_from_text_and_file(tmp_path, packed):
    model, obj, _ = packed
    source = textwrap.dedent('''
        import math

        SHIFT = 1.0

        def prepare(df):
            df["debt_log"] = np.log(df["debt_sum"] + SHIFT) * math.log(math.e)
            df["bal_filled"] = df["bal_avg_3m"].fillna(-1.0)
            return df
    ''')
    path = tmp_path / "prepare.py"
    path.write_text(source, encoding="utf-8")
    data = make_frame(300, seed=46)
    for value in (source, str(path)):
        other = pack_model(model, MODEL_FEATURES, MODEL_CATS, value, input_columns=RAW_COLUMNS, num_dtype="float32",
                           path=None, verbose=False)
        np.testing.assert_allclose(other.predict(data)["score"], obj.predict(data)["score"], rtol=0, atol=1e-12)


def test_pack_any_model_with_predict_proba():
    from sklearn.linear_model import LogisticRegression

    def fill(df):
        df["bal_avg_3m"] = df["bal_avg_3m"].fillna(0.0) / 1000.0
        return df

    frame = make_frame(1000, seed=47)
    features = ["bal_avg_3m", "dpd_sum_12m", "req_cnt_3m"]
    model = LogisticRegression(max_iter=500).fit(fill(frame.copy())[features].astype("float64"), frame["target_6m"])
    obj = pack_model(model, features, prepare=fill, sample=frame.head(200), path=None, verbose=False)
    assert obj.state["model_kind"] == "pickle"
    expected = model.predict_proba(fill(frame.copy())[features].astype("float64"))[:, 1]
    np.testing.assert_allclose(obj.predict(frame)["score"], expected, rtol=0, atol=1e-12)


OUTSIDE = 3


def uses_outside(df):
    return df.assign(debt_log=df["debt_sum"] + OUTSIDE, bal_filled=0.0)


def drops_rows(df):
    return df.assign(debt_log=0.0, bal_filled=0.0).head(10)


def loses_feature(df):
    return df


@pytest.mark.parametrize("func, kwargs, message", [
    (uses_outside, {}, "OUTSIDE"),
    (lambda df: df, {}, "lambda"),
    (drops_rows, {}, "строки"),
    (loses_feature, {}, "debt_log"),
    (prepare, {"sample": None}, "sample"),
    ("def prepare(df)\n    return df", {"input_columns": RAW_COLUMNS}, "ошибка в коде"),
    ("x = 1", {"input_columns": RAW_COLUMNS}, "def prepare"),
])
def test_pack_errors(packed, func, kwargs, message):
    model = packed[0]
    args = {"sample": make_frame(100, seed=48), "num_dtype": "float32", "path": None, "verbose": False, **kwargs}
    with pytest.raises(PackError, match=message):
        pack_model(model, MODEL_FEATURES, MODEL_CATS, func, **args)


def test_pack_rejects_bad_contract(packed):
    model, obj, _ = packed
    with pytest.raises(PackError, match="cat_features"):
        pack_model(model, MODEL_FEATURES, ["nope"], path=None, verbose=False)
    with pytest.raises(PackError, match="уже упакована"):
        pack_model(obj, MODEL_FEATURES, path=None, verbose=False)
    with pytest.raises(PackError, match="не модель"):
        pack_model(object(), MODEL_FEATURES, path=None, verbose=False)


def test_packed_reports_missing_columns(packed):
    with pytest.raises(ValueError, match="debt_sum"):
        packed[1].predict(make_frame(10, seed=49).drop(columns=["debt_sum"]))


BLOCK = "import sys\nsys.modules['scoring_kit'] = None\n"


def test_packed_loads_without_scoring_kit(tmp_path, packed):
    """В поде и в Spark нашего пакета нет: объект должен восстановиться сам и пережить повторный pickle."""
    model, _, path = packed
    data = make_frame(500, seed=50)
    data.to_csv(tmp_path / "data.csv", index=False)
    script = tmp_path / "load.py"
    script.write_text(BLOCK + textwrap.dedent(f'''
        import pickle
        import pandas as pd
        obj = pickle.load(open({str(path)!r}, "rb"))
        obj = pickle.loads(pickle.dumps(pickle.loads(pickle.dumps(obj))))
        obj.predict(pd.read_csv({str(tmp_path / "data.csv")!r})).to_csv({str(tmp_path / "out.csv")!r}, index=False)
    '''), encoding="utf-8")
    subprocess.run([sys.executable, str(script)], check=True, cwd=tmp_path)
    got = pd.read_csv(tmp_path / "out.csv")["score"].to_numpy()
    expected = expected_scores(model, pd.read_csv(tmp_path / "data.csv"))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_standalone_packer_file(tmp_path, packed):
    """`scoring packer`: упаковка в среде, где scoring-kit не установлен."""
    model, obj, _ = packed
    assert main(["packer", "-o", str(tmp_path / "scoring_pack.py")]) == 0
    joblib.dump(model, tmp_path / "model.pkl")
    data = make_frame(300, seed=51)
    data.to_csv(tmp_path / "data.csv", index=False)
    script = tmp_path / "use.py"
    script.write_text(BLOCK + textwrap.dedent(f'''
        import joblib
        import numpy as np
        import pandas as pd
        from scoring_pack import pack_model

        def prepare(df):
            df["debt_log"] = np.log1p(df["debt_sum"])
            df["bal_filled"] = df["bal_avg_3m"].fillna(-1.0)
            return df

        data = pd.read_csv("data.csv")
        obj = pack_model(joblib.load("model.pkl"), {MODEL_FEATURES!r}, {MODEL_CATS!r}, prepare, sample=data,
                         num_dtype="float32", path="packed.pkl", verbose=False)
        obj.predict(data).to_csv("out.csv", index=False)
    '''), encoding="utf-8")
    subprocess.run([sys.executable, str(script)], check=True, cwd=tmp_path)
    np.testing.assert_allclose(
        pd.read_csv(tmp_path / "out.csv")["score"], obj.predict(pd.read_csv(tmp_path / "data.csv"))["score"],
        rtol=0, atol=1e-12)
    assert load_packed(tmp_path / "packed.pkl").features == RAW_COLUMNS


def test_cli_describe(tmp_path, packed, capsys):
    assert main(["describe", str(packed[2])]) == 0
    assert "np.log1p" in capsys.readouterr().out
    joblib.dump(packed[0], tmp_path / "raw.pkl")
    assert main(["describe", str(tmp_path / "raw.pkl")]) == 1


# ---------------------------------------------------------------- конфиг


def test_spec_packed_rules(tmp_path):
    d = write_pipeline(tmp_path, DEMO, lambda raw: as_packed(raw["model"]))
    p = load_pipeline(d)
    assert p.model.packed and p.model.builtin and p.model.features == []
    with pytest.raises(PipelineError, match="уберите features"):
        load_pipeline(write_pipeline(tmp_path / "a", DEMO, lambda raw: raw["model"].update(packed=True, output_kind="proba")))
    with pytest.raises(PipelineError, match="перечислите фичи"):
        load_pipeline(write_pipeline(tmp_path / "b", DEMO, lambda raw: raw["model"].update(features=[], cat_features=[])))


# ---------------------------------------------------------------- сквозные прогоны


def test_gp_single_packed_matches_notebook(tmp_path, packed):
    model, _, path = packed
    d = write_pipeline(tmp_path, DEMO, lambda raw: as_packed(raw["model"]))
    data = make_frame(2500, seed=52)
    csv = tmp_path / "data.csv"
    data.to_csv(csv, index=False)
    result = run_debug(d, csv, [str(path)], tmp_path / "out")
    np.testing.assert_allclose(result.scored["score"].to_numpy(), expected_scores(model, pd.read_csv(csv)), rtol=0, atol=1e-12)
    assert len(result.scored) == 2500


def test_gp_config_and_registry_must_agree(tmp_path, packed, model_path):
    model, _, path = packed
    data = make_frame(200, seed=53)
    csv = tmp_path / "data.csv"
    data.to_csv(csv, index=False)
    as_is = write_pipeline(tmp_path / "a", DEMO, lambda raw: raw["model"].update(output_kind="proba"))
    with pytest.raises(Exception, match="упакована"):
        run_debug(as_is, csv, [str(path)], tmp_path / "out_a")
    flagged = write_pipeline(tmp_path / "b", DEMO, lambda raw: as_packed(raw["model"]))
    with pytest.raises(Exception, match="обычная модель"):
        run_debug(flagged, csv, [str(model_path)], tmp_path / "out_b")


def test_multi_packed_each_model_own_prepare(tmp_path):
    """Несколько моделей в одну витрину: у каждой свои фичи и своя обработка внутри упаковки."""
    def prep_a(df):
        df["debt_log"] = np.log1p(df["debt_sum"])
        return df

    def prep_b(df):
        df["load"] = df["dpd_sum_12m"] / (df["open_accnt_cnt"] + 1)
        df["bal_filled"] = df["bal_avg_3m"].fillna(0.0)
        return df

    specs = [
        (prep_a, ["segment", "debt_log", "dpd_sum_12m"], ["segment"]),
        (prep_b, ["load", "bal_filled", "refuse_flg"], ["refuse_flg"]),
        (None, ["segment", "req_cnt_3m", "debt_sum"], ["segment"]),
    ]
    sample = make_frame(300, seed=54)
    models, paths = [], []
    for i, (func, features, cats) in enumerate(specs):
        same = func or (lambda d: d)
        model = train(features, cats, seed=i, func=same)
        path = tmp_path / f"m{i}.pkl"
        pack_model(model, features, cats, func, sample=sample, num_dtype="float32", path=path, verbose=False)
        models.append((model, features, cats, same))
        paths.append(str(path))

    def edit(raw):
        raw["model_defaults"] = {k: v for k, v in raw["model_defaults"].items() if k in ("image", "requirements", "flavor")}
        raw["models"] = [as_packed(m) for m in raw["models"][:3]]
        columns = raw["sinks"][0]["columns"]
        raw["sinks"][0]["columns"] = {k: v for k, v in columns.items() if k != raw_last}

    raw_last = load_pipeline(MULTI).models[3].score_column
    d = write_pipeline(tmp_path, MULTI, edit)
    p = load_pipeline(d)
    data = make_frame(2000, seed=55)
    csv = tmp_path / "data.csv"
    data.to_csv(csv, index=False)
    result = run_debug(d, csv, paths, tmp_path / "out")
    back = pd.read_csv(csv)
    for m, (model, features, cats, func) in zip(p.models, models):
        np.testing.assert_allclose(
            result.scored[m.score_column].to_numpy(), expected_scores(model, back, features, cats, func), rtol=0, atol=1e-12)


def test_dlh_packed_matches_notebook(tmp_path, packed):
    model, _, path = packed
    d = write_pipeline(tmp_path, DLH, lambda raw: as_packed(raw["model"]))
    data = make_frame(1800, seed=56)
    csv = tmp_path / "data.csv"
    data.to_csv(csv, index=False)
    result = run_debug(d, csv, [str(path)], tmp_path / "out")
    np.testing.assert_allclose(result.scored["score"].to_numpy(), expected_scores(model, pd.read_csv(csv)), rtol=0, atol=1e-12)
    with pytest.raises(Exception, match="упакованной модели"):
        run_debug(d, csv, [str(joblib.dump(model, tmp_path / "raw.pkl")[0])], tmp_path / "out2")


def test_example_script_and_pipeline(tmp_path):
    """Пример из examples/demo_packed работает целиком: упаковка в «ноутбуке» -> процесс."""
    example = ROOT / "examples" / "demo_packed"
    env_root = str(ROOT)
    res = subprocess.run([sys.executable, str(example / "pack_model_example.py")], cwd=tmp_path, capture_output=True,
                         text=True, encoding="utf-8", env={**__import__("os").environ, "PYTHONPATH": env_root,
                                                           "PYTHONIOENCODING": "utf-8"})
    assert res.returncode == 0, res.stderr
    assert "совпадает с исходной моделью" in res.stdout
    obj = load_packed(tmp_path / "model_packed.pkl")
    assert obj.features == ["debt_amt", "dpd_days", "product_cd"]
    rng = np.random.default_rng(5)
    data = pd.DataFrame({"client_id": np.arange(300), "report_dt": "2026-09-01", "debt_amt": rng.gamma(2.0, 20000.0, 300),
                         "dpd_days": rng.integers(0, 120, 300).astype(float), "product_cd": rng.choice(["CC", "CL", "MT"], 300)})
    csv = tmp_path / "data.csv"
    data.to_csv(csv, index=False)
    result = run_debug(example, csv, [str(tmp_path / "model_packed.pkl")], tmp_path / "out")
    np.testing.assert_allclose(result.scored["score"], obj.predict(pd.read_csv(csv))["score"], rtol=0, atol=1e-12)
