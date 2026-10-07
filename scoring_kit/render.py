"""pipeline.yaml (+ predictor.py) → самодостаточный .py DAG.

Структура повторяет проверенное на платформе: колбэки и классы предикторов самодостаточны
(импорты внутри), данные между тасками передаются файлами через executor_config input/output.
Здесь — общая обвязка (шапка, default_args, алерты, документация, теги); граф строят рецепты.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from scoring_kit import recipes
from scoring_kit.blocks import WORK_IN, WORK_OUT
from scoring_kit.codegen import DagBuilder
from scoring_kit.spec import Pipeline, load_pipeline

READ_TASK = recipes.READ_TASK
INFERENCE_TASK = recipes.INFERENCE_TASK
PREDICTOR_CLASS = f"{INFERENCE_TASK}_predictor"

TIME_NOTIFIER = "from airflow_provider_time.notifications import TiMeNotifier"

__all__ = ["INFERENCE_TASK", "PREDICTOR_CLASS", "READ_TASK", "WORK_IN", "WORK_OUT", "render_dir", "render_pipeline"]


def _doc_md(p: Pipeline, source_name: str) -> str:
    lines = [f"### {p.dag_id}", ""]
    if p.description:
        lines += [p.description, ""]
    lines += [
        f"- **Владелец:** {p.owner}",
        f"- **Рецепт:** `{p.recipe}`, движок `{p.engine}`",
    ]
    if p.recipe == "fit_apply":
        lines.append(f"- **Калибровка:** `{p.calibrate.method}`, скор `{p.calibrate.score}` → `{p.calibrate.output}`")
    else:
        for m in p.all_models:
            lines.append(f"- **Модель** `{m.name or m.score_column}`: `{', '.join(m.mrid)}` → `{m.score_column}`")
    if p.wait_for:
        lines.append(f"- **Ждём актуальности:** {', '.join(f'`{t}`' for t in p.wait_for.tables)}")
    lines.append(f"- **Пишем в:** {', '.join(f'`{s.table}` ({s.mode})' for s in p.sinks)}")
    lines += ["", f"Сгенерировано scoring-kit из `{source_name or 'pipeline.yaml'}`. Руками не править."]
    return "\n".join(lines)


def render_pipeline(pipeline: Pipeline, predictor_source: Optional[str] = None, source_name: str = "") -> str:
    p = pipeline
    b = DagBuilder()
    b.need("import pendulum", "from airflow import DAG", "from datetime import timedelta")

    if p.recipe == "single_model":
        recipes.single_model(b, p, predictor_source)
    elif p.recipe == "multi_model":
        recipes.multi_model(b, p)
    else:
        recipes.fit_apply(b, p)

    default_args = [
        f"'owner': {p.owner!r}",
        f"'retries': {p.retries!r}",
        f"'retry_delay': timedelta(minutes={p.retry_delay_minutes!r})",
    ]
    if p.alerts:
        b.need(TIME_NOTIFIER)
        message = f"Скоринг {p.dag_id} упал. Владелец: {p.owner}." + (f" {p.alerts.message}" if p.alerts.message else "")
        notifier = f"TiMeNotifier(message={message!r}, recipients={list(p.alerts.recipients)!r})"
        default_args.append(f"'on_failure_callback': {notifier}")
        if p.alerts.on_retry:
            retry_message = f"Скоринг {p.dag_id}: повтор после ошибки."
            default_args.append(
                f"'on_retry_callback': TiMeNotifier(message={retry_message!r}, recipients={list(p.alerts.recipients)!r})"
            )

    tags = list(dict.fromkeys([*p.tags, *([p.domain] if p.domain else []), p.recipe, "scoring-kit"]))
    sd = p.start_date
    dag_open = [
        "with DAG(",
        f"    {p.dag_id!r},",
        f"    description={p.description!r},",
        f"    schedule={p.schedule!r},",
        f"    start_date=pendulum.datetime({sd.year}, {sd.month}, {sd.day}, tz={p.timezone!r}),",
        "    catchup=False,",
        "    max_active_runs=1,",
        f"    tags={tags!r},",
        f"    doc_md={_doc_md(p, source_name)!r},",
        "    default_args={",
        *[f"        {a}," for a in default_args],
        "    },",
        ") as dag:",
        "",
    ]
    origin = f" из {source_name}" if source_name else ""
    header = (
        f"# Сгенерировано scoring-kit{origin}.\n"
        "# Не редактируйте вручную: правьте pipeline.yaml / predictor.py и перегенерируйте."
    )
    code = b.render(header, dag_open)
    compile(code, f"{p.dag_id}.py", "exec")
    return code


def render_dir(pipeline_dir: Union[str, Path], shadow: bool = False) -> tuple[Pipeline, str]:
    pipeline_dir = Path(pipeline_dir)
    pipeline = load_pipeline(pipeline_dir)
    if shadow:
        pipeline = pipeline.as_shadow()
    source = None
    m = pipeline.model
    if m is not None and not m.builtin:
        source = (pipeline_dir / m.predictor).read_text(encoding="utf-8")
    name = f"{pipeline_dir.name}/pipeline.yaml"
    return pipeline, render_pipeline(pipeline, source, name)
