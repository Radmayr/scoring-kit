"""scoring publish: собрать полный набор DAG'ов для инстанса и опубликовать через mlc.

mlc публикует папку целиком и удаляет с инстанса всё, что было опубликовано через mlc и
чего нет в папке. Тестовый и боевой инстансы пишут в один Greenplum. Отсюда правила:

- test: теневые версии процессов с `shadow: true` (и из shadow.txt) + dags_manual/test/ (или dags_manual/*.py);
- prod: боевые версии всех процессов, кроме тех, что в shadow.txt, + dags_manual/prod/;
- пустой набор не публикуется никогда;
- перед публикацией — `mlc ... --check` и подтверждение.

Настройки инстансов — environments.yaml в корне репозитория процессов:

    test: {instance: collection-models-test, project: collectionanalytics}
    prod: {instance: collection-models-prod, project: collectionanalytics}

Опциональное поле `endpoint` задаёт mlcore API endpoint для окружения; если указано,
оно передаётся в mlc как `--endpoint`. Иначе mlc использует свой дефолтный (dev) endpoint.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from scoring_kit.render import render_dir
from scoring_kit.spec import PipelineError, load_pipeline

ENVIRONMENTS_FILE = "environments.yaml"
SHADOW_FILE = "shadow.txt"


@dataclass
class Plan:
    env: str
    instance: str
    project: str
    out: Path
    endpoint: Optional[str] = None  # mlcore API endpoint; если задан, передаётся в mlc --endpoint
    files: list[str] = field(default_factory=list)  # имена файлов в out/dags
    sources: dict[str, str] = field(default_factory=dict)  # файл -> откуда взят
    skipped: list[str] = field(default_factory=list)  # черновики, которые не публикуются


def load_environments(root: Path) -> dict:
    path = root / ENVIRONMENTS_FILE
    if not path.exists():
        raise PipelineError(
            f"Нет {path}. Создайте его, например:\n"
            "test: {instance: <тестовый инстанс>, project: <проект>}\n"
            "prod: {instance: <боевой инстанс>, project: <проект>}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for env, cfg in data.items():
        if not isinstance(cfg, dict) or not cfg.get("instance") or not cfg.get("project"):
            raise PipelineError(f"{path}: у окружения '{env}' нужны instance и project")
    return data


def pipeline_dirs(root: Path) -> list[Path]:
    base = root / "pipelines"
    return sorted(p for p in base.iterdir() if (p / "pipeline.yaml").exists()) if base.exists() else []


def shadow_dirs(root: Path) -> list[Path]:
    """Процессы в тени: `shadow: true` в их pipeline.yaml и, для совместимости, строки shadow.txt."""
    dirs = []
    path = root / SHADOW_FILE
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = (root / line.rstrip("/")).resolve()
            if not (d / "pipeline.yaml").exists():
                raise PipelineError(f"{SHADOW_FILE}: нет процесса {line}")
            dirs.append(d)
    for d in pipeline_dirs(root):
        if load_pipeline(d).shadow and d.resolve() not in dirs:
            dirs.append(d.resolve())
    return dirs


APPROVERS_FILE = "approvers.txt"


def approvers(root: Path) -> list[str]:
    """Логины тех, кому разрешено публиковать на боевой инстанс (по одному на строку)."""
    path = root / APPROVERS_FILE
    if not path.exists():
        return []
    lines = (line.split("#")[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line]


def check_approver(root: Path, user: str) -> None:
    allowed = approvers(root)
    if not allowed:
        raise PipelineError(f"{APPROVERS_FILE}: файл пуст или отсутствует — некому разрешено публиковать на прод")
    if user not in allowed:
        raise PipelineError(
            f"{user or '(логин не определён)'}: нет права публиковать на боевой инстанс. "
            f"Разрешено: {', '.join(allowed)} ({APPROVERS_FILE})"
        )


def _manual_files(root: Path, env: str) -> list[Path]:
    manual = root / "dags_manual"
    if (manual / env).is_dir():
        return sorted((manual / env).glob("*.py"))
    # Плоская папка (без test/ и prod/) — только для теста: это ваши отладочные DAG'и.
    if env == "test" and manual.is_dir():
        return sorted(manual.glob("*.py"))
    return []


def assemble(root: Path, env: str, out: Optional[Path] = None) -> Plan:
    root = root.resolve()
    environments = load_environments(root)
    if env not in environments:
        raise PipelineError(f"{ENVIRONMENTS_FILE}: нет окружения '{env}' (есть: {', '.join(environments)})")
    cfg = environments[env]
    out = (out or root / "publish" / env).resolve()
    plan = Plan(env=env, instance=cfg["instance"], project=cfg["project"], out=out, endpoint=cfg.get("endpoint"))

    shadow = shadow_dirs(root)
    if env == "test":
        targets = [(d, True) for d in shadow]
    else:
        targets = [(d.resolve(), False) for d in pipeline_dirs(root) if d.resolve() not in shadow]
    # Черновики (draft: true) не публикуются никуда.
    kept = []
    for d, as_shadow in targets:
        if load_pipeline(d).draft:
            plan.skipped.append(d.name)
        else:
            kept.append((d, as_shadow))
    targets = kept

    if out.exists():
        shutil.rmtree(out)
    dags = out / "dags"
    dags.mkdir(parents=True)

    for d, as_shadow in targets:
        pipeline, code = render_dir(d, shadow=as_shadow)
        name = f"{pipeline.dag_id}.py"
        if name in plan.sources:
            raise PipelineError(f"dag_id {pipeline.dag_id} повторяется: {plan.sources[name]} и {d.name}")
        (dags / name).write_text(code, encoding="utf-8")
        plan.sources[name] = f"{d.name}" + (" (тень)" if as_shadow else "")
    for f in _manual_files(root, env):
        if f.name in plan.sources:
            raise PipelineError(f"{f.name}: имя совпадает с DAG'ом фабрики ({plan.sources[f.name]})")
        shutil.copy2(f, dags / f.name)
        plan.sources[f.name] = f"dags_manual ({f.parent.name})" if f.parent.name in ("test", "prod") else "dags_manual"
    plan.files = sorted(plan.sources)
    if not plan.files:
        raise PipelineError(
            f"Для {env} нечего публиковать. Пустая публикация удалила бы все DAG'и с инстанса — отмена."
        )
    return plan


def _run(args: list[str]) -> int:
    print("$ " + " ".join(args), flush=True)
    try:
        return subprocess.run(args).returncode
    except FileNotFoundError:
        raise PipelineError("Не найдена команда mlc. Установите ML Core CLI или запускайте из CI.") from None


def _git(root: Path, *args: str) -> tuple[int, str]:
    try:
        res = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8")
    except OSError as e:
        return 1, str(e)
    return res.returncode, (res.stdout or "") + (res.stderr or "")


def in_ci() -> bool:
    return bool(os.environ.get("GITLAB_CI") or os.environ.get("CI"))


def check_local_copy(root: Path, git: Callable[..., tuple[int, str]] = _git) -> None:
    """Публикация с компьютера разрешена, только если копия совпадает с основной веткой в GitLab.

    mlc заменяет на инстансе весь набор DAG'ов: публикация из устаревшей или изменённой копии удалит
    процессы коллег или выложит то, чего нет в репозитории (и что пропадёт при следующей выкладке из CI).
    """
    code, out = git(root, "rev-parse", "--is-inside-work-tree")
    if code != 0 or out.strip() != "true":
        raise PipelineError(f"{root}: это не копия git-репозитория — непонятно, совпадает ли набор с GitLab")
    code, out = git(root, "status", "--porcelain", "--untracked-files=all")
    changed = [line[3:] for line in out.splitlines() if line.strip()]
    if code != 0 or changed:
        shown = ", ".join(changed[:5]) + (f" и ещё {len(changed) - 5}" if len(changed) > 5 else "")
        raise PipelineError(
            f"в копии есть изменения, которых нет в GitLab: {shown}. Отправьте их через merge request "
            "(или уберите) — иначе на инстанс уйдёт то, чего нет в репозитории"
        )
    code, out = git(root, "fetch", "--quiet", "origin")
    if code != 0:
        raise PipelineError(f"не удалось получить состояние GitLab (git fetch): {out.strip()[:200]}")
    code, head = git(root, "rev-parse", "HEAD")
    code2, ref = git(root, "rev-parse", "--abbrev-ref", "origin/HEAD")
    branch = ref.strip() if code2 == 0 and ref.strip() and "origin/" in ref else "origin/main"
    code3, remote = git(root, "rev-parse", branch)
    if code != 0 or code3 != 0:
        raise PipelineError(f"не удалось сравнить копию с {branch}")
    if head.strip() != remote.strip():
        code, behind = git(root, "rev-list", "--count", f"HEAD..{branch}")
        code, ahead = git(root, "rev-list", "--count", f"{branch}..HEAD")
        raise PipelineError(
            f"копия не совпадает с {branch} в GitLab (отстаёт на {behind.strip()} коммитов, "
            f"неотправленных коммитов: {ahead.strip()}). Сделайте git checkout main и git pull; "
            "свои изменения отправьте через merge request"
        )


def publish(
    root: Path,
    env: str,
    *,
    yes: bool = False,
    dry_run: bool = False,
    force_local: bool = False,
    run: Callable[[list[str]], int] = _run,
    ask: Callable[[str], str] = input,
    git: Callable[..., tuple[int, str]] = _git,
) -> Plan:
    # Публикация из CI собирает набор из основной ветки; с компьютера — только из копии, совпадающей с ней.
    if not dry_run and not in_ci():
        if force_local:
            print("ВНИМАНИЕ: --force-local — публикация без сверки с GitLab. Набор с этого компьютера заменит всё на инстансе.")
        else:
            try:
                check_local_copy(root, git)
            except PipelineError as e:
                raise PipelineError(
                    f"Публикация с компьютера отменена: {e}.\n"
                    "Обычный путь — выкладка из CI (merge request → слияние → deploy-test / deploy-prod)."
                ) from None
    plan = assemble(root, env)
    print(f"\nОкружение {env}: инстанс {plan.instance}, проект {plan.project}")
    print(f"Будет опубликовано {len(plan.files)} DAG'ов (папка {plan.out}):")
    for name in plan.files:
        print(f"  {name:45s} ← {plan.sources[name]}")
    if plan.skipped:
        print(f"Черновики (draft: true), не публикуются: {', '.join(plan.skipped)}")
    print("\nВсё, что опубликовано через mlc и чего нет в этом списке, исчезнет с инстанса.")
    if dry_run:
        print("Режим --dry-run: папка собрана, публикации не было.")
        return plan

    base = ["mlc", "airflow", "publish", plan.instance, "-p", plan.project, "-i", str(plan.out)]
    if plan.endpoint:
        base += ["--endpoint", plan.endpoint]
    if run(base + ["--check"]) != 0:
        raise PipelineError("mlc --check завершился с ошибкой — публикация отменена")
    if not yes:
        answer = ask(f"\nОпубликовать на {plan.instance}? Введите 'да': ").strip().lower()
        if answer not in ("да", "yes", "y", "д"):
            print("Отменено.")
            return plan
    if run(base + ["-y"]) != 0:
        raise PipelineError("mlc publish завершился с ошибкой")
    print(f"Опубликовано на {plan.instance}. Airflow подхватит изменения через 1–2 минуты.")
    return plan
