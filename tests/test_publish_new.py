"""scoring new (шаблоны всех рецептов), scoring publish (набор DAG'ов под инстанс), scoring schema."""

import json
import shutil

import pytest

from scoring_kit.cli import main
from scoring_kit.publish import assemble, publish
from scoring_kit.render import render_dir
from scoring_kit.spec import PipelineError, load_pipeline
from tests.conftest import ROOT

EXAMPLES = ROOT / "examples"


# ---------------------------------------------------------------- new


@pytest.mark.parametrize("recipe", ["single_model", "multi_model", "fit_apply", "dlh"])
def test_new_templates_are_valid(tmp_path, recipe):
    d = tmp_path / "pipelines" / f"my_{recipe}"
    assert main(["new", str(d), "--recipe", recipe]) == 0
    p = load_pipeline(d)
    assert p.dag_id == f"my_{recipe}"
    assert p.recipe == ("single_model" if recipe == "dlh" else recipe)
    render_dir(d)
    assert (d / "pipeline.yaml").read_text(encoding="utf-8").startswith("# yaml-language-server: $schema=")
    assert not (d / "predictor.py").exists()


def test_new_custom_predictor(tmp_path):
    d = tmp_path / "my_custom"
    assert main(["new", str(d), "--custom-predictor"]) == 0
    p = load_pipeline(d)
    assert not p.model.builtin
    assert "class inference_predictor(BasePredictor)" in render_dir(d)[1]


def test_new_rejects_bad_name_and_existing(tmp_path, capsys):
    assert main(["new", str(tmp_path / "My-Model")]) == 1
    assert "строчные" in capsys.readouterr().err
    d = tmp_path / "ok"
    d.mkdir()
    (d / "x").write_text("x")
    assert main(["new", str(d)]) == 1


# ---------------------------------------------------------------- schema


def test_schema(tmp_path):
    out = tmp_path / "pipeline.schema.json"
    assert main(["schema", "-o", str(out)]) == 0
    schema = json.loads(out.read_text(encoding="utf-8"))
    props = schema["properties"]
    assert {"recipe", "engine", "model_defaults", "sinks", "calibrate"} <= set(props)
    assert "fit_apply" in json.dumps(props["recipe"], ensure_ascii=False)


def test_model_defaults_field_does_not_leak(tmp_path):
    p = load_pipeline(EXAMPLES / "demo_multi")
    assert p.model_defaults is None
    assert p.models[0].image == "registry.example/mlops/lgbm:latest"


# ---------------------------------------------------------------- publish


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    for name in ("demo_scoring", "demo_multi", "demo_calibration", "demo_scoring_dlh"):
        shutil.copytree(EXAMPLES / name, root / "pipelines" / name)
    (root / "environments.yaml").write_text(
        "test: {instance: inst-test, project: proj}\nprod: {instance: inst-prod, project: proj}\n", encoding="utf-8"
    )
    (root / "shadow.txt").write_text("# в тени\npipelines/demo_multi\npipelines/demo_calibration/\n", encoding="utf-8")
    for env in ("test", "prod"):
        (root / "dags_manual" / env).mkdir(parents=True)
        (root / "dags_manual" / env / f"manual_{env}.py").write_text("# dag\n", encoding="utf-8")
    return root


def test_assemble_test_has_only_shadow_and_manual(repo):
    plan = assemble(repo, "test")
    assert plan.files == ["demo_calibration_shadow.py", "demo_multi_shadow.py", "manual_test.py"]
    assert plan.instance == "inst-test"
    assert sorted(p.name for p in (plan.out / "dags").iterdir()) == plan.files


def test_assemble_prod_excludes_shadowed(repo):
    plan = assemble(repo, "prod")
    assert plan.files == ["demo_scoring.py", "demo_scoring_dlh.py", "manual_prod.py"]


def test_flat_manual_dags_go_only_to_test(repo):
    shutil.rmtree(repo / "dags_manual")
    (repo / "dags_manual").mkdir()
    (repo / "dags_manual" / "introspect_dag.py").write_text("# dag\n", encoding="utf-8")
    assert "introspect_dag.py" in assemble(repo, "test").files
    assert "introspect_dag.py" not in assemble(repo, "prod").files


def test_empty_set_is_refused(repo):
    (repo / "shadow.txt").write_text("# пусто\n", encoding="utf-8")
    shutil.rmtree(repo / "dags_manual" / "test")
    with pytest.raises(PipelineError, match="нечего публиковать"):
        assemble(repo, "test")


@pytest.mark.parametrize(
    "files, message",
    [
        ({"environments.yaml": None}, "Нет"),
        ({"environments.yaml": "test: {instance: x}\n"}, "instance и project"),
        ({"shadow.txt": "pipelines/missing\n"}, "нет процесса"),
    ],
)
def test_config_errors(repo, files, message):
    for name, text in files.items():
        if text is None:
            (repo / name).unlink()
        else:
            (repo / name).write_text(text, encoding="utf-8")
    with pytest.raises(PipelineError, match=message):
        assemble(repo, "test")


def test_unknown_env(repo):
    with pytest.raises(PipelineError, match="нет окружения 'stage'"):
        assemble(repo, "stage")


def test_publish_check_confirm_publish(repo):
    calls = []
    publish(repo, "test", run=lambda a: calls.append(a) or 0, ask=lambda _: "да")
    assert [c[-1] for c in calls] == ["--check", "-y"]
    assert calls[0][:6] == ["mlc", "airflow", "publish", "inst-test", "-p", "proj"]


def test_publish_cancelled(repo):
    calls = []
    publish(repo, "test", run=lambda a: calls.append(a) or 0, ask=lambda _: "нет")
    assert [c[-1] for c in calls] == ["--check"]


def test_publish_stops_when_check_fails(repo):
    calls = []
    with pytest.raises(PipelineError, match="--check"):
        publish(repo, "test", run=lambda a: calls.append(a) or 1, ask=lambda _: "да")
    assert len(calls) == 1


def test_publish_yes_and_dry_run(repo):
    calls = []
    publish(repo, "prod", yes=True, run=lambda a: calls.append(a) or 0, ask=lambda _: pytest.fail("спросили"))
    assert len(calls) == 2
    calls.clear()
    publish(repo, "test", dry_run=True, run=lambda a: calls.append(a) or 0)
    assert calls == []


def test_draft_is_not_published(repo, capsys):
    path = repo / "pipelines" / "demo_multi" / "pipeline.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "draft: true\n", encoding="utf-8")
    plan = assemble(repo, "test")
    assert "demo_multi_shadow.py" not in plan.files and plan.skipped == ["demo_multi"]
    publish(repo, "test", dry_run=True)
    assert "Черновики" in capsys.readouterr().out
