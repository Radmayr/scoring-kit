"""Проверка merge request'а: боевые процессы и удаления меняет только владелец или approvers.txt."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scoring_kit.cli import main
from scoring_kit.guard import check_changes
from scoring_kit.publish import check_approver
from scoring_kit.spec import PipelineError

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def sh(cwd, *args):
    res = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, capture_output=True,
                         text=True, encoding="utf-8")
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


def set_fields(path, **fields):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for k, v in fields.items():
        if v is None:
            raw.pop(k, None)
        else:
            raw[k] = v
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


@pytest.fixture()
def repo(tmp_path):
    """prod_one — боевой процесс alice; shadow_one — процесс bob в тени; публиковать может boss."""
    root = tmp_path / "repo"
    for name, src, owner, shadow in (("prod_one", "demo_multi", "alice", None), ("shadow_one", "demo_calibration", "bob", True)):
        shutil.copytree(EXAMPLES / src, root / "pipelines" / name)
        set_fields(root / "pipelines" / name / "pipeline.yaml", owner=owner, shadow=shadow, dag_id=name)
    (root / "approvers.txt").write_text("# публикуют на прод\nboss\n", encoding="utf-8")
    (root / ".gitlab-ci.yml").write_text("stages: [check]\n", encoding="utf-8")
    sh(tmp_path, "init", "-b", "main", str(root))
    sh(root, "add", "-A")
    sh(root, "commit", "-m", "init")
    return root, sh(root, "rev-parse", "HEAD")


def commit(root, message="change"):
    sh(root, "add", "-A")
    sh(root, "commit", "-m", message)


def edit(root, name, **fields):
    set_fields(root / "pipelines" / name / "pipeline.yaml", **fields)


def test_anyone_may_add_process_and_edit_shadow(repo):
    root, base = repo
    shutil.copytree(EXAMPLES / "demo_scoring", root / "pipelines" / "new_one")
    edit(root, "shadow_one", description="правка тени")
    commit(root)
    assert check_changes(root, base, "carol") == []


def test_prod_process_only_owner_or_approver(repo):
    root, base = repo
    edit(root, "prod_one", description="правка боевого")
    commit(root)
    problems = check_changes(root, base, "carol")
    assert problems == ["изменение боевого процесса prod_one (владелец alice)"]
    assert check_changes(root, base, "alice") == []
    assert check_changes(root, base, "boss") == []


def test_moving_prod_process_to_shadow_counts_as_change(repo):
    root, base = repo
    edit(root, "prod_one", shadow=True)
    commit(root)
    assert check_changes(root, base, "carol") == ["изменение боевого процесса prod_one (владелец alice)"]


def test_deleting_any_process_only_owner_or_approver(repo):
    root, base = repo
    shutil.rmtree(root / "pipelines" / "shadow_one")
    commit(root)
    assert check_changes(root, base, "carol") == ["удаление процесса shadow_one (владелец bob)"]
    assert check_changes(root, base, "bob") == []


def test_cannot_make_yourself_owner_or_approver_in_same_change(repo):
    root, base = repo
    edit(root, "prod_one", owner="carol", description="теперь моё")
    (root / "approvers.txt").write_text("boss\ncarol\n", encoding="utf-8")
    commit(root)
    problems = check_changes(root, base, "carol")
    assert "изменение боевого процесса prod_one (владелец alice)" in problems
    assert "изменение служебного файла approvers.txt" in problems


def test_service_files_only_approver(repo):
    root, base = repo
    (root / ".gitlab-ci.yml").write_text("stages: []\n", encoding="utf-8")
    commit(root)
    assert check_changes(root, base, "alice") == ["изменение служебного файла .gitlab-ci.yml"]
    assert check_changes(root, base, "boss") == []


def test_cli_check_changes_explains(repo, capsys):
    root, base = repo
    edit(root, "prod_one", description="правка")
    commit(root)
    assert main(["check-changes", "--base", base, "--user", "carol", "--root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "prod_one" in err and "Run pipeline" in err and "boss" in err
    assert main(["check-changes", "--base", base, "--user", "alice", "--root", str(root)]) == 0
    with pytest.raises(PipelineError, match="нет базовой версии"):
        check_changes(root, "0" * 40, "alice")


def test_prod_publish_by_owner_of_prod_process(repo):
    root, _ = repo
    check_approver(root, "boss")
    with pytest.raises(PipelineError, match="нет права"):
        check_approver(root, "alice")                         # без --allow-owners — только approvers.txt
    check_approver(root, "alice", allow_owners=True)          # владелец боевого процесса
    with pytest.raises(PipelineError, match="владельцы боевых процессов: alice"):
        check_approver(root, "bob", allow_owners=True)         # у bob только процесс в тени
