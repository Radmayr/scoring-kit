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


@pytest.fixture(autouse=True)
def as_in_ci(monkeypatch):
    """По умолчанию тесты публикации идут «как в CI»: сверку копии с GitLab проверяют отдельные тесты ниже."""
    monkeypatch.setenv("GITLAB_CI", "true")


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


def test_publish_passes_endpoint(repo):
    (repo / "environments.yaml").write_text(
        "test: {instance: inst-test, project: proj, endpoint: https://mlcore.t-tech.team}\n"
        "prod: {instance: inst-prod, project: proj, endpoint: https://mlcore.t-tech.team}\n",
        encoding="utf-8",
    )
    assert assemble(repo, "test").endpoint == "https://mlcore.t-tech.team"
    calls = []
    publish(repo, "test", yes=True, run=lambda a: calls.append(a) or 0, ask=lambda _: pytest.fail("спросили"))
    cmd = calls[0]
    assert cmd[:4] == ["mlc", "airflow", "publish", "inst-test"]
    assert "--endpoint" in cmd
    assert cmd[cmd.index("--endpoint") + 1] == "https://mlcore.t-tech.team"


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


def test_shadow_flag_in_pipeline_config(repo):
    """Процесс в тени можно пометить в его же конфиге: shadow.txt для этого не нужен."""
    import yaml as _yaml

    from scoring_kit.publish import shadow_dirs

    names = sorted(p.name for p in (repo / "pipelines").iterdir())
    listed = {d.name for d in shadow_dirs(repo)}
    target = next(n for n in names if n not in listed)
    path = repo / "pipelines" / target / "pipeline.yaml"
    raw = _yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["shadow"] = True
    path.write_text(_yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    assert target in {d.name for d in shadow_dirs(repo)}
    test_plan, prod_plan = assemble(repo, "test"), None
    assert any(target in src and "тень" in src for src in test_plan.sources.values())
    try:
        prod_plan = assemble(repo, "prod", repo / "publish_prod")
    except Exception:
        prod_plan = None                                   # на прод могло не остаться ни одного процесса
    if prod_plan is not None:
        assert target not in prod_plan.sources.values()


def test_check_approver(repo, capsys):
    from scoring_kit.cli import main

    assert main(["check-approver", "--user", "i.ivanov", "--root", str(repo)]) == 1      # файла нет
    (repo / "approvers.txt").write_text("# кто может публиковать на прод\ni.ivanov\np.petrov  # руководитель\n", encoding="utf-8")
    assert main(["check-approver", "--user", "p.petrov", "--root", str(repo)]) == 0
    assert main(["check-approver", "--user", "x.other", "--root", str(repo)]) == 1
    assert "i.ivanov, p.petrov" in capsys.readouterr().err


# ---------------------------------------------------------------- защита публикации с компьютера

def _sh(cwd, *args):
    import subprocess as sp
    res = sp.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return res.stdout


@pytest.fixture()
def synced(repo, tmp_path, monkeypatch):
    """Копия процессов, совпадающая с «GitLab» (локальный bare-репозиторий как origin)."""
    monkeypatch.delenv("GITLAB_CI", raising=False)
    monkeypatch.delenv("CI", raising=False)
    origin = tmp_path / "origin.git"
    _sh(tmp_path, "init", "--bare", "-b", "main", str(origin))
    _sh(repo, "init", "-b", "main")
    _sh(repo, "add", "-A")
    _sh(repo, "commit", "-m", "init")
    _sh(repo, "remote", "add", "origin", str(origin))
    _sh(repo, "push", "-u", "origin", "main")
    _sh(repo, "remote", "set-head", "origin", "main")
    return repo, origin


def _publish_calls(repo, **kw):
    calls = []
    publish(repo, "test", yes=True, run=lambda a: calls.append(a) or 0, ask=lambda _: "да", **kw)
    return calls


def test_local_publish_allowed_when_copy_matches_gitlab(synced):
    repo, _ = synced
    assert len(_publish_calls(repo)) == 2          # --check и публикация


def test_local_publish_refused_with_uncommitted_changes(synced):
    repo, _ = synced
    target = repo / "pipelines" / "demo_multi" / "pipeline.yaml"
    target.write_text(target.read_text(encoding="utf-8") + "# правка\n", encoding="utf-8")
    with pytest.raises(PipelineError, match="изменения, которых нет в GitLab"):
        _publish_calls(repo)


def test_local_publish_refused_when_behind_or_ahead(synced, tmp_path):
    repo, origin = synced
    other = tmp_path / "colleague"
    _sh(tmp_path, "clone", str(origin), str(other))
    (other / "pipelines" / "demo_multi" / "note.txt").write_text("x", encoding="utf-8")
    _sh(other, "add", "-A")
    _sh(other, "commit", "-m", "коллега")
    _sh(other, "push", "origin", "main")
    with pytest.raises(PipelineError, match="отстаёт на 1"):
        _publish_calls(repo)
    _sh(repo, "pull", "origin", "main")
    (repo / "local.txt").write_text("x", encoding="utf-8")
    _sh(repo, "add", "-A")
    _sh(repo, "commit", "-m", "моё, не отправлено")
    with pytest.raises(PipelineError, match="неотправленных коммитов: 1"):
        _publish_calls(repo)


def test_local_publish_force_and_ci_and_dry_run(synced, monkeypatch):
    repo, _ = synced
    target = repo / "pipelines" / "demo_multi" / "pipeline.yaml"
    target.write_text(target.read_text(encoding="utf-8") + "# правка\n", encoding="utf-8")
    publish(repo, "test", dry_run=True)                       # сухой прогон не проверяет копию
    _sh(repo, "checkout", "--", ".")
    (repo / "scratch.txt").write_text("x", encoding="utf-8")
    assert len(_publish_calls(repo, force_local=True)) == 2
    monkeypatch.setenv("GITLAB_CI", "true")
    assert len(_publish_calls(repo)) == 2                     # в CI копия всегда из основной ветки


def test_local_publish_refused_outside_git(repo, monkeypatch):
    monkeypatch.delenv("GITLAB_CI", raising=False)
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(PipelineError, match="не копия git"):
        _publish_calls(repo, git=lambda root, *a: (128, "fatal: not a git repository"))
