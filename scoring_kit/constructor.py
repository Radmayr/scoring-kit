"""Конструктор скоринга: страница-форма, которая собирает pipeline.yaml без программирования.

Страница одна и без сервера: её можно открыть файлом, положить в GitLab Pages или любой
статический хостинг. Командные настройки (образы, сервис Greenplum, ссылка на GitLab)
подставляются в неё при сборке, поэтому реальные адреса не хранятся в scoring-kit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

import yaml

TEMPLATE = Path(__file__).parent / "web" / "constructor.html"
START, END = "/*TEAM_PRESETS_START*/", "/*TEAM_PRESETS_END*/"
KINDS = ("model", "job", "spark")
HEAD = ('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n')


class PresetsError(ValueError):
    pass


def load_presets(path: Union[str, Path]) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PresetsError(f"{path}: ожидается словарь настроек")
    return raw


def check_presets(presets: dict) -> dict:
    """Проверяет и нормализует пресеты: gp_service, gitlab_new_file_url, domains, images."""
    allowed = {"gp_service", "gitlab_new_file_url", "domains", "images"}
    extra = set(presets) - allowed
    if extra:
        raise PresetsError(f"неизвестные поля пресетов: {sorted(extra)}; есть {sorted(allowed)}")
    images, ids = [], set()
    for i, img in enumerate(presets.get("images") or []):
        for field in ("id", "kind", "label", "image"):
            if not img.get(field):
                raise PresetsError(f"images[{i}]: нужно поле {field}")
        if img["kind"] not in KINDS:
            raise PresetsError(f"images[{i}].kind: одно из {KINDS} (model — модели, job — калибровка, spark — DLH)")
        if img["id"] in ids or img["id"] == "custom":
            raise PresetsError(f"images[{i}].id '{img['id']}': повтор или зарезервированное имя")
        ids.add(img["id"])
        images.append({"id": str(img["id"]), "kind": img["kind"], "label": str(img["label"]),
                       "image": str(img["image"]), "requirements": [str(r) for r in img.get("requirements") or []]})
    return {
        "gp_service": str(presets.get("gp_service") or ""),
        "gitlab_new_file_url": str(presets.get("gitlab_new_file_url") or ""),
        "domains": [str(d) for d in presets.get("domains") or []],
        "images": images,
    }


def presets_from_pipelines(dirs: Iterable[Union[str, Path]]) -> dict:
    """Собирает пресеты из уже работающих процессов: их образы, сервис Greenplum и направления."""
    from scoring_kit.spec import load_pipeline

    images: list[dict] = []
    seen: set = set()
    services: list[str] = []
    domains: list[str] = []

    def add(kind: str, image: Optional[str], requirements: list, source: str):
        if not image:
            return
        key = (kind, image, tuple(requirements))
        if key in seen:
            return
        seen.add(key)
        short = image.rstrip("/").split("/")[-1]
        images.append({"id": f"{kind}_{len(images) + 1}", "kind": kind, "label": f"{short} (как в {source})",
                       "image": image, "requirements": list(requirements)})

    for d in dirs:
        p = load_pipeline(d)
        name = p.dag_id
        for m in p.all_models:
            if p.engine != "dlh":
                add("model", m.image, m.requirements, name)
        if p.job is not None:
            add("job", p.job.image, p.job.requirements, name)
        if p.dlh is not None:
            add("spark", p.dlh.image, [], name)
        if p.engine != "dlh" and p.gp_service not in services:
            services.append(p.gp_service)
        if p.domain and p.domain not in domains:
            domains.append(p.domain)
    return {"gp_service": services[0] if services else "", "domains": domains, "images": images}


def merge_presets(base: dict, extra: dict) -> dict:
    """Пресеты из файла главнее; образы из процессов дописываются, если таких ещё нет."""
    out = dict(extra)
    out.update({k: v for k, v in base.items() if v})
    images = list(base.get("images") or [])
    have = {(i["kind"], i["image"], tuple(i.get("requirements") or [])) for i in images}
    ids = {i["id"] for i in images}
    for img in extra.get("images") or []:
        key = (img["kind"], img["image"], tuple(img["requirements"]))
        if key not in have:
            new = dict(img)
            while new["id"] in ids:
                new["id"] += "_"
            ids.add(new["id"])
            images.append(new)
    out["images"] = images
    return out


def build_constructor(presets: Optional[dict] = None) -> str:
    """Готовая страница: шаблон + командные пресеты, с doctype, чтобы открывалась файлом."""
    text = TEMPLATE.read_text(encoding="utf-8")
    if presets is not None:
        team = check_presets(presets)
        a, b = text.index(START) + len(START), text.index(END)
        text = text[:a] + "\nvar TEAM = " + json.dumps(team, ensure_ascii=False, indent=2) + ";\n" + text[b:]
    return HEAD + text + "\n</html>\n"
