"""scoring publish: собрать полный набор DAG'ов для инстанса и опубликовать через mlc.

mlc публикует папку целиком и удаляет с инстанса всё, что было опубликовано через mlc и
чего нет в папке. Тестовый и боевой инстансы пишут в один Greenplum. Отсюда правила:

- test: теневые версии процессов из shadow.txt + dags_manual/test/ (или dags_manual/*.py);
- prod: боевые версии всех процессов, кроме тех, что в shadow.txt, + dags_manual/prod/;
- пустой набор не публикуется никогда;
- перед публикацией — `mlc ... --check` и подтверждение.

Настройки инстансов — environments.yaml в корне репозитория процессов:

    test: {instance: collection-models-test, project: collectionanalytics}
    prod: {instance: collection-models-prod, project: collectionanalytics}
"""

from __future__ import annotations

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
    path = root / SHADOW_FILE
    if not path.exists():
        return []
    dirs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = (root / line.rstrip("/")).resolve()
        if not (d / "pipeline.yaml").exists():
            raise PipelineError(f"{SHADOW_FILE}: нет процесса {line}")
        dirs.append(d)
    return dirs


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
    out = (out or root / "publish").resolve()
    plan = Plan(env=env, instance=cfg["instance"], project=cfg["project"], out=out)

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


def publish(
    root: Path,
    env: str,
    *,
    yes: bool = False,
    dry_run: bool = False,
    run: Callable[[list[str]], int] = _run,
    ask: Callable[[str], str] = input,
) -> Plan:
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
