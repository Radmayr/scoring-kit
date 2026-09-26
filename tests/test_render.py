import ast
import builtins
import symtable

import pytest

from scoring_kit.render import render_dir, render_pipeline
from scoring_kit.spec import load_pipeline
from scoring_kit.stubs import load_dag
from tests.conftest import DEMO
from tests.test_spec import make_dir

BUILTINS = set(dir(builtins))


def _free_globals(table: symtable.SymbolTable) -> set:
    names = set()
    if table.get_type() == "function":
        names |= {n for n in table.get_globals()}
    for child in table.get_children():
        names |= _free_globals(child)
    return names


def _top_level_objects(code: str):
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            yield node.name, ast.get_source_segment(code, node)


@pytest.fixture(scope="module")
def demo():
    pipeline, code = render_dir(DEMO)
    return pipeline, code, load_dag(code)


def test_callbacks_and_predictor_are_self_contained(demo):
    """В под уходит только исходник функции/класса — внешних имён быть не должно."""
    _, code, _ = demo
    objects = dict(_top_level_objects(code))
    assert set(objects) == {"read_source_callback", "inference_predictor", "write_demo_scores_fresh_callback"}
    for name, src in objects.items():
        table = symtable.symtable(src, name, "exec")
        free = _free_globals(table) - BUILTINS
        assert not free, f"{name} ссылается на внешние имена {free}"


def test_graph(demo):
    _, _, dag = demo
    t = dag.tasks
    assert set(t) == {
        "wait_source",
        "read_source",
        "inference",
        "write_demo_scores_fresh_stg",
        "write_demo_scores_fresh_load",
        "write_demo_scores_fresh_harmonize",
        "write_demo_scores_fresh_actualize",
    }
    up = {k: {u.task_id for u in v.upstream} for k, v in t.items()}
    assert up["wait_source"] == set()
    assert up["read_source"] == {"wait_source"}
    assert up["inference"] == {"read_source"}
    assert up["write_demo_scores_fresh_stg"] == {"inference"}
    assert up["write_demo_scores_fresh_load"] == {"write_demo_scores_fresh_stg"}
    assert up["write_demo_scores_fresh_harmonize"] == {"write_demo_scores_fresh_load"}
    assert up["write_demo_scores_fresh_actualize"] == {"write_demo_scores_fresh_harmonize"}


def test_dag_arguments(demo):
    pipeline, _, dag = demo
    assert dag.dag_id == "demo_scoring"
    assert dag.kwargs["schedule"] == "30 4 * * *"
    assert dag.kwargs["catchup"] is False
    assert dag.kwargs["default_args"]["owner"] == "your.login"
    assert dag.tasks["wait_source"].kwargs["retries"] == 0


def test_file_handoff_between_tasks(demo):
    """Выход одного таска смонтирован во вход следующего по тем же путям."""
    _, _, dag = demo
    read = dag.tasks["read_source"].kwargs
    inf = dag.tasks["inference"].kwargs
    stg = dag.tasks["write_demo_scores_fresh_stg"].kwargs
    assert read["executor_config"]["output"] == [{"src": "/work/output", "name": "output"}]
    assert inf["executor_config"]["input"] == [{"src": "read_source/output", "dst": "/work/input"}]
    assert inf["input_df_path"] == "/work/input/data.csv"
    assert inf["output_df_path"] == "/work/output/data.csv"
    assert stg["executor_config"]["input"] == [{"src": "inference/output", "dst": "/work/input"}]
    assert stg["table"] == "sandbox.demo_scores_fresh_stg"
    assert stg["if_exists"] == "replace"


def test_operator_arguments_match_spec(demo):
    pipeline, _, dag = demo
    read = dag.tasks["read_source"].kwargs
    inf = dag.tasks["inference"].kwargs
    assert read["query"].strip() == pipeline.source.query
    assert read["gp_service"] == "vrcl" and read["mode"] == "dal"
    assert inf["mrid"] == pipeline.model.mrid
    assert inf["requirements"] == pipeline.model.requirements
    assert inf["batch_size"] == 100000
    assert dag.tasks["write_demo_scores_fresh_stg"].kwargs["columns_types"] == pipeline.sinks[0].columns


def test_replace_is_truncate_insert_in_transaction(demo):
    _, _, dag = demo
    q = dag.tasks["write_demo_scores_fresh_load"].kwargs["query"]
    assert q.startswith("begin;")
    assert "truncate table sandbox.demo_scores_fresh;" in q
    assert "from sandbox.demo_scores_fresh_stg;" in q
    assert q.rstrip().endswith("commit;")
    harm = dag.tasks["write_demo_scores_fresh_harmonize"].kwargs["query"]
    assert harm == "select public.tcs_harmonize_grants('sandbox.demo_scores_fresh')"


def test_append_with_service_columns(tmp_path):
    d = make_dir(
        tmp_path,
        sinks__0__mode="append",
        sinks__0__add_scored_at=True,
        sinks__0__add_model_version=True,
        sinks__0__harmonize=False,
        sinks__0__actualize=False,
    )
    _, code = render_dir(d)
    dag = load_dag(code)
    q = dag.tasks["write_demo_scores_fresh_load"].kwargs["query"]
    assert "truncate" not in q
    assert "scored_at, model_mrid)" in q
    assert "now(), 'demo_tenant/demo_model/0.0.1'" in q
    assert "write_demo_scores_fresh_harmonize" not in dag.tasks


def test_no_wait_and_several_sinks(tmp_path):
    second = {
        "table": "sandbox.demo_scores_hist",
        "mode": "append",
        "columns": {"client_id": "bigint", "score": "numeric"},
        "add_scored_at": True,
    }
    pipeline = load_pipeline(make_dir(tmp_path, wait_for=...))
    pipeline = pipeline.model_copy(
        update={"sinks": [*pipeline.sinks, type(pipeline.sinks[0]).model_validate(second)]}
    )
    code = render_pipeline(pipeline, (DEMO / "predictor.py").read_text(encoding="utf-8"))
    assert "GreenplumTablesWaitSensor" not in code
    dag = load_dag(code)
    ups = {k: {u.task_id for u in v.upstream} for k, v in dag.tasks.items()}
    assert ups["read_source"] == set()
    assert ups["write_demo_scores_fresh_stg"] == {"inference"}
    assert ups["write_demo_scores_hist_stg"] == {"inference"}


def test_shadow_render():
    pipeline, code = render_dir(DEMO, shadow=True)
    dag = load_dag(code)
    assert dag.dag_id == "demo_scoring_shadow"
    assert "write_demo_scores_fresh_shadow_stg" in dag.tasks
    stg = dag.tasks["write_demo_scores_fresh_shadow_stg"].kwargs
    assert stg["table"] == "sandbox.demo_scores_fresh_shadow_stg"
    assert "write_demo_scores_fresh_shadow_harmonize" in dag.tasks
    assert "write_demo_scores_fresh_shadow_actualize" not in dag.tasks


def test_introspect_dag_loads():
    from tests.conftest import ROOT

    code = (ROOT / "tools" / "introspect_dag.py").read_text(encoding="utf-8")
    dag = load_dag(code)
    assert set(dag.tasks) == {"introspect", "gp_version"}


def test_multiline_query_is_readable(demo):
    _, code, _ = demo
    assert "query='''select *\nfrom sandbox.demo_features_fresh'''" in code
