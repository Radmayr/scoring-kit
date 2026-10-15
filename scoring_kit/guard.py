"""Проверка merge request'а: кто может менять боевые процессы и удалять процессы.

Правило:
- новые процессы и правки процессов в тени (`shadow: true`) может вносить любой участник;
- изменить боевой процесс, удалить любой процесс или поменять служебные файлы репозитория может
  только владелец этого процесса (поле `owner` в версии до изменения) или участник из approvers.txt
  (тоже по версии до изменения: добавить себя в список и сразу воспользоваться этим нельзя).

Проверка ловит ошибки, а не злой умысел: человек с правом записи может поправить и сам файл CI.
Гарантию дают права в GitLab (защищённая ветка) и на инстансе.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

import yaml

from scoring_kit.spec import PipelineError

APPROVERS_FILE = "approvers.txt"
SHADOW_FILE = "shadow.txt"
# Файлы, от которых зависит, что и куда выкладывается, и кто может публиковать.
PROTECTED_FILES = (".gitlab-ci.yml", APPROVERS_FILE, "environments.yaml", "requirements.txt", SHADOW_FILE)

Git = Callable[..., tuple]


def owners_of(cfg: Optional[dict]) -> list[str]:
    """Логины из поля owner: один или несколько через запятую или пробел."""
    raw = (cfg or {}).get("owner") or ""
    return [x for x in re.split(r"[,;\s]+", str(raw)) if x]


def parse_approvers(text: Optional[str]) -> list[str]:
    lines = (line.split("#")[0].strip() for line in (text or "").splitlines())
    return [line for line in lines if line]


def _show(root: Path, git: Git, base: str, path: str) -> Optional[str]:
    code, out = git(root, "show", f"{base}:{path}")
    return out if code == 0 else None


def _ensure_base(root: Path, git: Git, base: str) -> None:
    code, _ = git(root, "cat-file", "-e", f"{base}^{{commit}}")
    if code != 0:
        git(root, "fetch", "--quiet", "origin", base)
        code, _ = git(root, "cat-file", "-e", f"{base}^{{commit}}")
    if code != 0:
        raise PipelineError(f"нет базовой версии {base}: не с чем сравнить изменения")


def check_changes(root: Path, base: str, user: str, git: Optional[Git] = None) -> list[str]:
    """Список изменений, которые этому пользователю вносить нельзя (пустой — всё в порядке)."""
    if git is None:
        from scoring_kit.publish import _git as git
    root = Path(root)
    _ensure_base(root, git, base)
    if user and user in parse_approvers(_show(root, git, base, APPROVERS_FILE)):
        return []

    code, out = git(root, "diff", "--name-only", base, "HEAD")
    if code != 0:
        raise PipelineError(f"не удалось получить список изменений: {out.strip()[:200]}")
    paths = [line.strip() for line in out.splitlines() if line.strip()]
    base_shadow = {line.strip().rstrip("/").split("/")[-1] for line in parse_approvers(_show(root, git, base, SHADOW_FILE))}

    problems = []
    folders = sorted({p.split("/")[1] for p in paths if p.startswith("pipelines/") and p.count("/") >= 2})
    for folder in folders:
        before = _show(root, git, base, f"pipelines/{folder}/pipeline.yaml")
        if before is None:
            continue                                   # новый процесс: может добавить любой
        cfg = yaml.safe_load(before) or {}
        owners = owners_of(cfg)
        if user and user in owners:
            continue
        who = f"владелец {', '.join(owners)}" if owners else "владелец не указан"
        in_prod = not (cfg.get("shadow") or folder in base_shadow) and not cfg.get("draft")
        if not (root / "pipelines" / folder / "pipeline.yaml").exists():
            problems.append(f"удаление процесса {folder} ({who})")
        elif in_prod:
            problems.append(f"изменение боевого процесса {folder} ({who})")
    for path in paths:
        if path in PROTECTED_FILES:
            problems.append(f"изменение служебного файла {path}")
    return problems


def explain(problems: list[str], user: str, approvers: list[str]) -> str:
    listed = "\n".join(f"  - {p}" for p in problems)
    allowed = ", ".join(approvers) if approvers else "(список пуст)"
    return (
        f"{user or '(логин не определён)'}: в этом merge request есть изменения, которые может сделать только "
        f"владелец процесса или участник из {APPROVERS_FILE} ({allowed}):\n{listed}\n"
        "Как быть: попросите владельца или участника из списка нажать в merge request «Run pipeline» — "
        "проверка пройдёт от его имени; или пусть он сделает это изменение сам."
    )
