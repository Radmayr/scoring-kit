import shutil

import pytest
import yaml

from scoring_kit.spec import PipelineError, load_pipeline
from tests.conftest import DEMO


def make_dir(tmp_path, **patch):
    """Копия демо-пайплайна с изменёнными полями (patch: путь через __ -> значение)."""
    d = tmp_path / "p"
    shutil.copytree(DEMO, d)
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


def test_demo_is_valid():
    p = load_pipeline(DEMO)
    assert p.dag_id == "demo_scoring"
    assert p.model.mrid == ["demo_tenant/demo_model/0.0.1"]
    assert p.sinks[0].stg_table == "sandbox.demo_scores_fresh_stg"


def test_mrid_string_becomes_list(tmp_path):
    p = load_pipeline(make_dir(tmp_path, model__mrid="a/b/0.0.1"))
    assert p.model.mrid == ["a/b/0.0.1"]


def test_column_types_are_normalized(tmp_path):
    cols = {"client_id": "int ", "segment": "varchar ", "score": " numeric"}
    p = load_pipeline(make_dir(tmp_path, sinks__0__columns=cols))
    assert p.sinks[0].columns == {"client_id": "int", "segment": "varchar", "score": "numeric"}


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"model__cat_features": ["nope"]}, "не входят в features"),
        ({"sinks__0__columns": {"client_id": "int"}}, "нет колонки скора"),
        ({"sinks__0__table": "no_schema"}, "schema.table"),
        ({"sinks__0__table": "Sandbox.X"}, "schema.table"),
        ({"model__features": ["a", "a"]}, "повторяются"),
        ({"model__unknown_field": 1}, "unknown_field"),
        ({"transport": "parquet"}, "transport"),
        ({"owner": ...}, "owner"),
        ({"dag_id": "bad id"}, "dag_id"),
        ({"source__query": "   "}, "пустой"),
        ({"model__score_range": [1, 0]}, "левая граница"),
    ],
)
def test_invalid_configs(tmp_path, patch, message):
    with pytest.raises(PipelineError, match=message):
        load_pipeline(make_dir(tmp_path, **patch))


def test_service_column_clash(tmp_path):
    cols = {"client_id": "bigint", "score": "numeric", "scored_at": "timestamp"}
    with pytest.raises(PipelineError, match="добавляются автоматически"):
        load_pipeline(make_dir(tmp_path, sinks__0__columns=cols, sinks__0__add_scored_at=True))


def test_missing_predictor_file(tmp_path):
    with pytest.raises(PipelineError, match="не найден файл предиктора"):
        load_pipeline(make_dir(tmp_path, model__predictor="other.py"))


def test_shadow_copy():
    p = load_pipeline(DEMO).as_shadow()
    assert p.dag_id == "demo_scoring_shadow"
    assert p.sinks[0].table == "sandbox.demo_scores_fresh_shadow"
    assert "shadow" in p.tags
