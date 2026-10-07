"""Каталог процессов: CATALOG.md со списком, владельцами, моделями, таблицами и зависимостями.

Зависимость A → B появляется, если B ждёт (wait_for) таблицу, в которую пишет A.
"""

from __future__ import annotations

from pathlib import Path

from scoring_kit.spec import Pipeline, PipelineError, load_pipeline


def _schedule_msk(p: Pipeline) -> str:
    if not p.schedule:
        return "вручную"
    parts = p.schedule.split()
    if p.timezone.upper() == "UTC" and len(parts) == 5 and parts[0].isdigit() and parts[1].isdigit():
        hour = (int(parts[1]) + 3) % 24
        if parts[2:] == ["*", "*", "*"]:
            return f"ежедневно {hour:02d}:{int(parts[0]):02d} МСК"
    return f"`{p.schedule}` ({p.timezone})"


def _models(p: Pipeline) -> str:
    if p.recipe == "fit_apply":
        return f"калибровка `{p.calibrate.method}` ({p.calibrate.score} → {p.calibrate.output})"
    return "<br>".join(f"`{mrid}` → {m.score_column}" for m in p.all_models for mrid in m.mrid)


def _sources(p: Pipeline) -> list[str]:
    tables = list(p.wait_for.tables) if p.wait_for else []
    if p.source is not None and p.source.table:
        tables.append(p.source.table)
    return list(dict.fromkeys(tables))


def build_catalog(dirs: list[Path]) -> str:
    pipelines: list[tuple[Path, Pipeline]] = []
    errors: list[str] = []
    for d in dirs:
        try:
            pipelines.append((d, load_pipeline(d)))
        except PipelineError as e:
            errors.append(f"- `{d}`: {str(e).splitlines()[0]}")

    writers = {s.table: p.dag_id for _, p in pipelines for s in p.sinks}
    edges = sorted({
        (writers[t], p.dag_id) for _, p in pipelines for t in _sources(p) if t in writers and writers[t] != p.dag_id
    })

    lines = [
        "# Каталог процессов скоринга",
        "",
        "Генерируется командой `scoring catalog`. Не править вручную.",
        "",
        "| Процесс | Рецепт | Владелец | Расписание | Модели | Источники | Приёмники |",
        "|---|---|---|---|---|---|---|",
    ]
    for d, p in sorted(pipelines, key=lambda x: (x[1].domain or "", x[1].dag_id)):
        sources = "<br>".join(f"`{t}`" for t in _sources(p)) or "—"
        sinks = "<br>".join(f"`{s.table}` ({s.mode})" for s in p.sinks)
        recipe = p.recipe + (f" / {p.engine}" if p.engine != "gp" else "")
        lines.append(
            f"| **{p.dag_id}**<br>{p.description} | {recipe} | {p.owner} | {_schedule_msk(p)} | "
            f"{_models(p)} | {sources} | {sinks} |"
        )
    if edges:
        lines += ["", "## Зависимости", "", "```mermaid", "flowchart LR"]
        lines += [f"    {a} --> {b}" for a, b in edges]
        lines += ["```"]
    # Модель → процессы: «где используется модель X».
    usage: dict[str, list[str]] = {}
    for _, p in pipelines:
        for m in p.all_models:
            for mrid in m.mrid:
                usage.setdefault(mrid, []).append(p.dag_id)
    if usage:
        lines += ["", "## Модели", "", "| mrid | Процессы |", "|---|---|"]
        lines += [f"| `{k}` | {', '.join(sorted(v))} |" for k, v in sorted(usage.items())]
    if errors:
        lines += ["", "## Невалидные конфиги", "", *errors]
    return "\n".join(lines) + "\n"
