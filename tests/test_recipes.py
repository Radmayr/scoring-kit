"""Рецепты multi_model, fit_apply и движок dlh: граф, валидация, паритет с текущими процессами."""

import shutil

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import yaml

from scoring_kit.debug import operator_predictor, run_debug
from scoring_kit.render import render_dir
from scoring_kit.spec import PipelineError, load_pipeline
from scoring_kit.stubs import load_dag
from tests.conftest import CAT_FEATURES, DEMO, FEATURES, ROOT, make_frame
from tests.legacy import legacy_calibration, legacy_multi_scores, legacy_scores, make_lgbm_ready
from tests.test_render import BUILTINS, _free_globals, _top_level_objects

MULTI = ROOT / "examples" / "demo_multi"
CALIB = ROOT / "examples" / "demo_calibration"
DLH = ROOT / "examples" / "demo_scoring_dlh"


def patched(tmp_path, src, **patch):
    d = tmp_path / src.name
    shutil.copytree(src, d)
    raw = yaml.safe_load((d / "pipeline.yaml").read_text(encoding="utf-8"))
    for key, value in patch.items():
        node = raw
        *path, last = key.split("__")
        for part in path:
            node = node[int(part)] if isinstance(node, list) else node[part]
        if value is ...:
            del node[last]
        else:
            node[last] = value
    (d / "pipeline.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return d


def upstream(dag):
    return {k: {u.task_id for u in v.upstream} for k, v in dag.tasks.items()}


@pytest.mark.parametrize("example", [MULTI, CALIB, DLH])
def test_generated_code_is_self_contained(example):
    """Колбэки и классы уходят в поды исходником — ссылок на имена модуля быть не должно."""
    _, code = render_dir(example)
    for name, src in _top_level_objects(code):
        import symtable

        free = _free_globals(symtable.symtable(src, name, "exec")) - BUILTINS
        assert not free, f"{name}: внешние имена {free}"


def test_alerts_on_failure():
    _, code = render_dir(MULTI)
    dag = load_dag(code)
    cb = dag.kwargs["default_args"]["on_failure_callback"]
    assert cb.recipients == ["#scoring-alerts"]
    assert "demo_multi" in cb.message
    assert dag.kwargs["max_active_runs"] == 1
    assert "multi_model" in dag.kwargs["tags"] and "collection" in dag.kwargs["tags"]
    assert "demo_tenant/demo_d1/0.0.1" in dag.kwargs["doc_md"]


# ---------------------------------------------------------------- multi_model


def test_multi_one_job_graph():
    _, code = render_dir(MULTI)
    dag = load_dag(code)
    up = upstream(dag)
    assert up["read_source"] == {"wait_source"}
    assert up["inference"] == {"read_source"}
    assert up["write_demo_multi_scores_fresh_stg"] == {"inference"}
    inf = dag.tasks["inference"].kwargs
    assert inf["mrid"] == [f"demo_tenant/demo_d{i}/0.0.1" for i in range(1, 5)]
    assert inf["flavor"] == "4cpu-32ram"
    cls = operator_predictor(code, "inference_predictor")
    assert [c["score_column"] for c in cls.contracts] == ["score_10", "score_30", "score_90", "score_90_6m"]


def test_multi_groups_by_runtime(tmp_path):
    d = patched(tmp_path, MULTI, models__2__image="registry.example/other:1", models__3__image="registry.example/other:1")
    _, code = render_dir(d)
    dag = load_dag(code)
    up = upstream(dag)
    assert dag.tasks["inference_1"].kwargs["mrid"] == ["demo_tenant/demo_d1/0.0.1", "demo_tenant/demo_d2/0.0.1"]
    assert dag.tasks["inference_2"].kwargs["mrid"] == ["demo_tenant/demo_d3/0.0.1", "demo_tenant/demo_d4/0.0.1"]
    assert up["merge_scores"] == {"inference_1", "inference_2"}
    assert up["write_demo_multi_scores_fresh_stg"] == {"merge_scores"}


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"models": [{"name": "a", "mrid": "t/a/1", "image": "i", "output_kind": "proba", "features": ["segment", "x"], "score_column": "score_10"}]}, "хотя бы две"),
        ({"models__0__name": ...}, "нужно имя"),
        ({"models__0__cat_features": []}, "разные типы"),
        ({"models__1__score_column": "score_10"}, "повторяются"),
        ({"sinks__0__columns": {"client_id": "bigint", "score_10": "numeric"}}, "нет колонок скоров"),
    ],
)
def test_multi_validation(tmp_path, patch, message):
    with pytest.raises(PipelineError, match=message):
        load_pipeline(patched(tmp_path, MULTI, **patch))


@pytest.fixture(scope="module")
def multi_models(tmp_path_factory):
    """Четыре модели на разных подмножествах фичей, обученные как в ноутбуках (make_lgbm_ready)."""
    p = load_pipeline(MULTI)
    train = make_frame(4000, seed=11)
    d = tmp_path_factory.mktemp("multi_models")
    out = {}
    for i, m in enumerate(p.models):
        X, _ = make_lgbm_ready(train[m.features], cat_cols=[c for c in m.cat_features if c in m.features])
        model = lgb.LGBMClassifier(n_estimators=30 + 10 * i, num_leaves=7 + i, random_state=i)
        model.fit(X[m.features], train["target_6m"])
        path = d / f"{m.name}.pkl"
        joblib.dump(model, path)
        out[m.name] = (path, model, m)
    return out


def test_multi_scores_match_legacy(tmp_path, multi_models):
    data = make_frame(3000, seed=12)
    csv = tmp_path / "data.csv"
    data.to_csv(csv, index=False)
    result = run_debug(MULTI, csv, [str(v[0]) for v in multi_models.values()], tmp_path / "out")
    legacy = legacy_multi_scores(
        {m.score_column: (model, m.features) for _, model, m in multi_models.values()}, csv, ["segment"]
    )
    for _, _, m in multi_models.values():
        np.testing.assert_array_equal(result.scored[m.score_column].to_numpy(), legacy[m.score_column].to_numpy())
    written = result.to_write["sandbox.demo_multi_scores_fresh"]
    assert list(written.columns) == list(load_pipeline(MULTI).sinks[0].columns)


def test_multi_two_jobs_merge(tmp_path, multi_models):
    d = patched(tmp_path, MULTI, models__2__image="registry.example/other:1", models__3__image="registry.example/other:1")
    data = make_frame(1500, seed=13)
    csv = tmp_path / "data.csv"
    data.to_csv(csv, index=False)
    by_name = {f"{k}": str(v[0]) for k, v in multi_models.items()}
    result = run_debug(d, csv, [f"{k}={v}" for k, v in by_name.items()], tmp_path / "out")
    one_job = run_debug(MULTI, csv, list(by_name.values()), tmp_path / "out1")
    for col in ["score_10", "score_30", "score_90", "score_90_6m"]:
        np.testing.assert_allclose(result.scored[col], one_job.scored[col], rtol=0, atol=1e-12)
    assert len(result.scored) == 1500


# ---------------------------------------------------------------- single_model со встроенным предиктором


def test_builtin_single_matches_predictor_file(tmp_path, model_path):
    d = patched(tmp_path, DEMO, model__output_kind="proba")
    (d / "predictor.py").unlink()
    data = make_frame(2000, seed=21)
    csv = tmp_path / "data.csv"
    data.to_csv(csv, index=False)
    builtin = run_debug(d, csv, [model_path], tmp_path / "a")
    expected = legacy_scores(joblib.load(model_path), csv, FEATURES, CAT_FEATURES, batch_size=1000)
    np.testing.assert_array_equal(builtin.scored["score"].to_numpy(), expected.to_numpy())


# ---------------------------------------------------------------- fit_apply


def calib_data(seed=0):
    rng = np.random.default_rng(seed)
    cohorts = [f"2025-{m:02d}" for m in range(1, 10)]
    products = ["CCR", "CUR", "REF", "MTG", "MTF", "MTB"]
    n = 12000
    dev = pd.DataFrame({
        "account_id": np.arange(n),
        "report_dt": "2026-09-01",
        "cohort": rng.choice(cohorts, n),
        "base_score": rng.uniform(0.01, 0.6, n).round(6),
        "product_cd": rng.choice(products, n),
    })
    dev["base_target"] = (rng.random(n) < dev["base_score"] * 0.8).astype(int)
    m = 3000
    apply = pd.DataFrame({
        "account_id": np.arange(m) + 100000,
        "report_dt": "2026-09-02",
        "cohort": rng.choice(cohorts[3:] + ["2025-10"], m),   # 2025-10 нет в dev → без калибровки
        "base_score": rng.uniform(0.01, 0.6, m).round(6),
        "product_cd": rng.choice(products + ["NEW"], m),      # NEW нет в сегментах → без калибровки
    })
    return dev, apply


def test_fit_apply_graph():
    _, code = render_dir(CALIB)
    dag = load_dag(code)
    up = upstream(dag)
    assert up["fit_apply"] == {"read_dev", "read_apply"}
    fa = dag.tasks["fit_apply"].kwargs
    assert fa["mrid"] == [] and fa["batch_size"] is None
    assert {i["dst"] for i in fa["executor_config"]["input"]} == {"/work/dev", "/work/input"}
    assert up["write_demo_scores_calib_fresh_stg"] == {"fit_apply"}
    assert up["write_demo_calib_report_hist_stg"] == {"fit_apply"}
    assert "report.csv" in code


def test_calibration_matches_legacy(tmp_path):
    dev, apply = calib_data()
    dev.to_csv(tmp_path / "dev.csv", index=False)
    apply.to_csv(tmp_path / "apply.csv", index=False)
    result = run_debug(CALIB, {"dev": tmp_path / "dev.csv", "apply": tmp_path / "apply.csv"}, [], tmp_path / "out")
    segments = load_pipeline(CALIB).calibrate.segments
    legacy = legacy_calibration(
        dev, apply, segments, "base_score", "base_target", "cohort", ["account_id", "report_dt"], tmp_path / "legacy.csv"
    )
    got = result.scored.set_index("account_id")["base_score_calib"]
    exp = legacy.set_index("account_id")["base_score_calib"].reindex(got.index)
    assert got.isna().sum() == exp.isna().sum() > 0  # некалиброванные строки — те же
    np.testing.assert_array_equal(got.to_numpy(), exp.to_numpy())
    report = result.to_write["sandbox.demo_calib_report_hist"]
    assert {"segment", "cohort", "brier_raw", "brier_calibrated"} <= set(report.columns)
    ok = report[report["status"] == "ok"]
    assert len(ok) and (ok["brier_calibrated"] <= ok["brier_raw"] + 0.01).all()


def test_calibration_copy_score(tmp_path):
    d = patched(tmp_path, CALIB, calibrate__uncalibrated="copy_score")
    dev, apply = calib_data(1)
    dev.to_csv(tmp_path / "dev.csv", index=False)
    apply.to_csv(tmp_path / "apply.csv", index=False)
    result = run_debug(d, {"dev": tmp_path / "dev.csv", "apply": tmp_path / "apply.csv"}, [], tmp_path / "out")
    s = result.scored
    assert s["base_score_calib"].notna().all()
    new = s["product_cd"] == "NEW"
    np.testing.assert_array_equal(s.loc[new, "base_score_calib"], s.loc[new, "base_score"])


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"calibrate__train_offsets": [1]}, "отрицательные"),
        ({"apply__key": []}, "ключ apply"),
        ({"sinks__1__columns": {"segment": "varchar", "oops": "int"}}, "нет колонок"),
        ({"source": {"query": "select 1"}}, "не используются"),
    ],
)
def test_fit_apply_validation(tmp_path, patch, message):
    with pytest.raises(PipelineError, match=message):
        load_pipeline(patched(tmp_path, CALIB, **patch))


# ---------------------------------------------------------------- engine: dlh


def test_dlh_graph_with_query():
    _, code = render_dir(DLH)
    dag = load_dag(code)
    up = upstream(dag)
    assert type(dag.tasks["wait_source"]).__name__ == "DLHTablesWaitSensor"
    assert up["prepare_source"] == {"wait_source"}
    assert up["check_source"] == {"prepare_source"}
    assert up["inference"] == {"check_source"}
    assert up["check_result"] == {"inference"}
    inf = dag.tasks["inference"].kwargs
    meta = inf["models_meta"][0]
    assert meta.key_columns == ["client_id", "account_id", "report_dt", "segment"]
    assert meta.output_columns == ["score"]
    assert meta.output_types["score"].type_name == "DoubleType"
    assert inf["source_table"] == "sandbox.demo_scoring_dlh_src"
    assert inf["target_table"] == "sandbox.demo_scores_dlh"
    prep = dag.tasks["prepare_source"].kwargs["query"]
    assert prep[0] == "drop table if exists sandbox.demo_scoring_dlh_src"
    assert prep[1].startswith("create table sandbox.demo_scoring_dlh_src as")


def test_dlh_graph_with_table(tmp_path):
    d = patched(tmp_path, DLH, source__query=..., source__table="sandbox.ready_features", dlh__staging_table=...)
    dag = load_dag(render_dir(d)[1])
    assert "prepare_source" not in dag.tasks
    assert dag.tasks["inference"].kwargs["source_table"] == "sandbox.ready_features"


def test_dlh_shadow():
    p, code = render_dir(DLH, shadow=True)
    inf = load_dag(code).tasks["inference"].kwargs
    assert inf["target_table"] == "sandbox.demo_scores_dlh_shadow"
    assert inf["source_table"] == "sandbox.demo_scoring_dlh_src_shadow"


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"dlh__staging_table": ...}, "staging_table"),
        ({"model__output_kind": ...}, "output_kind"),
        ({"sinks__0__mode": "append"}, "replace"),
        ({"sinks__0__columns__processed_dttm": "timestamp"}, "processed_dttm"),
        ({"dlh": ...}, "нужны поля"),
        ({"recipe": "multi_model"}, "нужны поля"),
    ],
)
def test_dlh_validation(tmp_path, patch, message):
    with pytest.raises(PipelineError, match=message):
        load_pipeline(patched(tmp_path, DLH, **patch))


def test_dlh_scores_match_legacy(tmp_path, model_path):
    """Сырая модель упаковывается в бандл; скоры как у старого кода на lightgbm 3.3.5."""
    data = make_frame(2500, seed=31)
    csv = tmp_path / "data.csv"
    data.to_csv(csv, index=False)
    result = run_debug(DLH, csv, [model_path], tmp_path / "out")
    expected = legacy_scores(joblib.load(model_path), csv, FEATURES, CAT_FEATURES, batch_size=10**9)
    np.testing.assert_array_equal(result.scored["score"].to_numpy(), expected.to_numpy())
    assert list(result.scored.columns) == ["client_id", "account_id", "report_dt", "segment", "score", "processed_dttm"]


def test_dlh_duplicate_keys_stop(tmp_path, model_path):
    data = make_frame(500, seed=32)
    data = pd.concat([data, data.head(2)])
    csv = tmp_path / "dups.csv"
    data.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="не уникален"):
        run_debug(DLH, csv, [model_path], tmp_path / "out")
