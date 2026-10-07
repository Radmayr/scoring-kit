"""Рецепты: yaml → граф блоков.

Рецепт не знает операторов, только блоки (scoring_kit.blocks). Каждая функция получает
DagBuilder и провалидированный Pipeline и возвращает ничего: результат — код в билдере.
"""

from __future__ import annotations

from typing import Optional

from scoring_kit import blocks, runtime
from scoring_kit.codegen import DagBuilder
from scoring_kit.predictor_source import build_predictor_class
from scoring_kit.predictors import builtin_predictor_class, calibration_predictor_class
from scoring_kit.spec import DLH_PROCESSED_COLUMN, Model, Pipeline, PipelineError

INFERENCE_TASK = "inference"
READ_TASK = "read_source"


def required_source_columns(p: Pipeline) -> list[str]:
    """Что должно прийти из выборки: фичи, ключ и колонки приёмников, которые не создаёт инференс."""
    produced = set(p.output_columns)
    passthrough = [c for s in p.sinks if s.data == "result" for c in s.columns if c not in produced]
    features = [f for m in p.all_models for f in m.features]
    return list(dict.fromkeys(features + p.source.checks.unique_key + passthrough))


def _write_all(b: DagBuilder, p: Pipeline, upstream: str) -> None:
    for sink in p.sinks:
        b.edge(upstream, blocks.gp_write(b, p, sink, upstream))


def _start(b: DagBuilder, p: Pipeline, sensor) -> Optional[str]:
    return sensor(b, p) if p.wait_for else None


# ---------------------------------------------------------------- single_model


def single_model(b: DagBuilder, p: Pipeline, predictor_source: Optional[str]) -> None:
    if p.engine == "dlh":
        return _single_model_dlh(b, p)
    m = p.model
    first = _start(b, p, blocks.gp_sensor)
    read = blocks.gp_read(
        b, p, READ_TASK, p.source.sql, p.source.flavor, p.source.checks.min_rows,
        required_source_columns(p), p.source.checks.unique_key,
    )
    if first:
        b.edge(first, read)
    class_name = f"{INFERENCE_TASK}_predictor"
    if m.builtin:
        class_code = builtin_predictor_class([m], class_name)
    else:
        if predictor_source is None:
            raise PipelineError("single_model: нет исходника predictor.py")
        class_code = build_predictor_class(predictor_source, m, class_name)
    infer = blocks.mlc_infer(
        b, p, INFERENCE_TASK, class_code, class_name, m.mrid, m.image, m.requirements, m.flavor, m.batch_size,
        [(read, blocks.WORK_IN)], f"{blocks.WORK_IN}/{blocks.DATA_FILE}",
    )
    b.edge(read, infer)
    _write_all(b, p, infer)


def _single_model_dlh(b: DagBuilder, p: Pipeline) -> None:
    m, sink, dlh = p.model, p.sinks[0], p.dlh
    first = _start(b, p, blocks.dlh_sensor)
    prev = first
    if p.source.query:
        source_table = dlh.staging_table
        prep = blocks.dlh_prepare(b, p.source.query, source_table)
        if prev:
            b.edge(prev, prep)
        prev = prep
    else:
        source_table = p.source.table

    key = p.source.checks.unique_key
    required = required_source_columns(p)
    dup_sql = (
        f"(select count(*) from (select {', '.join(key)} from {source_table} "
        f"group by {', '.join(key)} having count(*) > 1) d)"
        if key
        else "0"
    )
    check_src = blocks.dlh_check(
        b,
        "check_source",
        "select\n"
        f"    (select count(*) from {source_table}) as n_rows,\n"
        f"    {dup_sql} as dup_keys,\n"
        # упадёт с понятной ошибкой Trino, если какой-то колонки нет
        f"    (select count(*) from (select {', '.join(required)} from {source_table} limit 1) c) as columns_ok",
        runtime.sk_dlh_check_source,
        f"min_rows={p.source.checks.min_rows!r}",
    )
    if prev:
        b.edge(prev, check_src)

    # Всё, что пишется, кроме скора, оператор переносит из входа «как есть» (key_columns).
    key_columns = [c for c in sink.columns if c != m.score_column]
    infer = blocks.dlh_infer(
        b, INFERENCE_TASK, key_columns, m.score_column, m.mrid[0], source_table, sink.table,
        dlh.image, dlh.max_executors, dlh.max_wait_seconds,
    )
    b.edge(check_src, infer)

    score_range = tuple(m.score_range) if m.score_range is not None else None
    check_out = blocks.dlh_check(
        b,
        "check_result",
        "select\n"
        f"    count(*) as n_rows,\n"
        f"    count({m.score_column}) as n_score,\n"
        f"    min({m.score_column}) as min_score,\n"
        f"    max({m.score_column}) as max_score,\n"
        f"    (select count(*) from {source_table}) as n_source\n"
        f"from {sink.table}",
        runtime.sk_dlh_check_target,
        f"score_range={score_range!r}",
    )
    b.edge(infer, check_out)


# ---------------------------------------------------------------- multi_model


def runtime_groups(models: list[Model]) -> list[list[Model]]:
    """Модели с одинаковым окружением и ресурсами — один job, порядок сохраняется."""
    groups: dict[tuple, list[Model]] = {}
    for m in models:
        groups.setdefault(m.runtime_key, []).append(m)
    return list(groups.values())


def multi_model(b: DagBuilder, p: Pipeline) -> None:
    first = _start(b, p, blocks.gp_sensor)
    read = blocks.gp_read(
        b, p, READ_TASK, p.source.sql, p.source.flavor, p.source.checks.min_rows,
        required_source_columns(p), p.source.checks.unique_key,
    )
    if first:
        b.edge(first, read)
    groups = runtime_groups(p.models)
    if len(groups) > 1 and not p.source.checks.unique_key:
        raise PipelineError(
            "multi_model: модели в разных окружениях считаются отдельными job'ами и сливаются по ключу — "
            "укажите source.checks.unique_key"
        )
    infer_tasks = []
    for i, group in enumerate(groups, start=1):
        task_id = INFERENCE_TASK if len(groups) == 1 else f"{INFERENCE_TASK}_{i}"
        class_name = f"{task_id}_predictor"
        g0 = group[0]
        infer = blocks.mlc_infer(
            b, p, task_id, builtin_predictor_class(group, class_name), class_name,
            [m.mrid[0] for m in group], g0.image, g0.requirements, g0.flavor, g0.batch_size,
            [(read, blocks.WORK_IN)], f"{blocks.WORK_IN}/{blocks.DATA_FILE}",
        )
        b.edge(read, infer)
        infer_tasks.append((infer, [m.score_column for m in group]))
    last = infer_tasks[0][0]
    if len(infer_tasks) > 1:
        last = blocks.py_merge(
            b, p, "merge_scores", [t for t, _ in infer_tasks], p.source.checks.unique_key,
            [cols for _, cols in infer_tasks], p.source.flavor,
        )
        for t, _ in infer_tasks:
            b.edge(t, last)
    _write_all(b, p, last)


# ---------------------------------------------------------------- fit_apply


def fit_apply(b: DagBuilder, p: Pipeline) -> None:
    first = _start(b, p, blocks.gp_sensor)
    cfg = p.calibrate
    dev_required = list(dict.fromkeys([cfg.score, cfg.target, cfg.cohort_column, *cfg.segments]))
    produced = {cfg.output}
    passthrough = [c for s in p.sinks if s.data == "result" for c in s.columns if c not in produced]
    apply_required = list(dict.fromkeys([cfg.score, cfg.cohort_column, *cfg.segments, *p.apply.key, *passthrough]))
    dev = blocks.gp_read(b, p, "read_dev", p.dev.query, p.dev.flavor, p.dev.min_rows, dev_required, [])
    apply = blocks.gp_read(b, p, "read_apply", p.apply.query, p.apply.flavor, p.apply.min_rows, apply_required, p.apply.key)
    if first:
        b.edge(first, dev)
        b.edge(first, apply)
    task_id = "fit_apply"
    class_name = f"{task_id}_predictor"
    class_code = calibration_predictor_class(
        cfg, f"/work/dev/{blocks.DATA_FILE}", f"{blocks.WORK_OUT}/{blocks.REPORT_FILE}", class_name
    )
    job = p.job
    infer = blocks.mlc_infer(
        b, p, task_id, class_code, class_name, [], job.image, job.requirements, job.flavor, None,
        [(dev, "/work/dev"), (apply, blocks.WORK_IN)], f"{blocks.WORK_IN}/{blocks.DATA_FILE}",
    )
    b.edge(dev, infer)
    b.edge(apply, infer)
    _write_all(b, p, infer)


__all__ = [
    "DLH_PROCESSED_COLUMN",
    "INFERENCE_TASK",
    "READ_TASK",
    "fit_apply",
    "multi_model",
    "required_source_columns",
    "runtime_groups",
    "single_model",
]
