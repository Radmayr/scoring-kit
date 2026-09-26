"""pipeline.yaml + predictor.py -> самодостаточный .py DAG.

Структура повторяет то, что генерировал старый инструмент и что проверено на
платформе: колбэки и класс предиктора самодостаточны (импорты внутри), данные
между тасками передаются файлами через executor_config input/output.
"""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path
from typing import Callable, Union

from scoring_kit import runtime
from scoring_kit.codefmt import lit as _lit
from scoring_kit.predictor_source import build_predictor_class
from scoring_kit.spec import Pipeline, load_pipeline
from scoring_kit.sql import actualize_sql, harmonize_sql, load_sql

WORK_OUT = "/work/output"
WORK_IN = "/work/input"
READ_TASK = "read_source"
INFERENCE_TASK = "inference"
PREDICTOR_CLASS = "inference_predictor"


def _helpers(*funcs: Callable) -> str:
    return "\n".join(inspect.getsource(f) for f in funcs)


def _func(name: str, args: str, helpers: tuple, body: str) -> str:
    inner = _helpers(*helpers) + "\n" + textwrap.dedent(body).strip() + "\n"
    return f"def {name}({args}):\n" + textwrap.indent(inner, "    ")


def _call(target: str, cls: str, kwargs: dict, indent: int) -> str:
    """target = Cls(k=v, ...); значения — готовые строки кода."""
    pad = " " * (indent + 4)
    args = "".join(f"{pad}{k}={v},\n" for k, v in kwargs.items())
    return f"{' ' * indent}{target} = {cls}(\n{args}{' ' * indent})"


def render_pipeline(pipeline: Pipeline, predictor_source: str, source_name: str = "") -> str:
    p = pipeline
    data = p.data_file
    # Всё, что должно прийти из выборки: фичи, ключ и колонки приёмников, которые predict не создаёт.
    # Проверка на входе: иначе ошибка о недостающей колонке всплывёт только после скоринга, на записи.
    passthrough = [
        c
        for s in p.sinks
        for c in s.columns
        if c != p.model.score_column and c not in p.model.output_columns
    ]
    required = list(dict.fromkeys(p.model.features + p.source.checks.unique_key + passthrough))

    read_cb = _func(
        f"{READ_TASK}_callback",
        "df",
        (runtime.sk_normalize_decimals, runtime.sk_check_source, runtime.sk_write_frame),
        f"""
        df = sk_normalize_decimals(df)
        sk_check_source(
            df,
            min_rows={p.source.checks.min_rows!r},
            required_columns={required!r},
            unique_key={p.source.checks.unique_key!r},
        )
        sk_write_frame(df, {f"{WORK_OUT}/{data}"!r})
        """,
    )

    predictor_cls = build_predictor_class(predictor_source, p.model, PREDICTOR_CLASS)

    write_cbs = []
    for s in p.sinks:
        write_cbs.append(
            _func(
                f"write_{s.short_name}_callback",
                "",
                (runtime.sk_read_frame, runtime.sk_conform_to_columns),
                f"""
                df = sk_read_frame({f"{WORK_IN}/{data}"!r})
                return sk_conform_to_columns(df, {s.columns!r})
                """,
            )
        )

    blocks: list[str] = []
    origin = f" из {source_name}" if source_name else ""
    blocks.append(
        f"# Сгенерировано scoring-kit{origin}.\n"
        "# Не редактируйте вручную: правьте pipeline.yaml / predictor.py и перегенерируйте.\n"
        "from datetime import timedelta\n\n"
        "import pendulum\n"
        "from airflow import DAG\n"
        "from airflow.utils.task_group import TaskGroup\n"
        "from airflow_provider_greenplum.operators.greenplum import DataframeToGreenplumOperator\n"
        "from airflow_provider_greenplum.operators.greenplum import GreenplumExecuteOperator\n"
        "from airflow_provider_greenplum.operators.greenplum import GreenplumToDataframeOperator\n"
        + (
            "from airflow_provider_greenplum.sensors.greenplum import GreenplumTablesWaitSensor\n"
            if p.wait_for
            else ""
        )
        + "from airflow_provider_inference.operators.inference import BasePredictor, BatchInferenceOperator"
    )
    blocks.append(read_cb)
    blocks.append(predictor_cls.rstrip("\n"))
    blocks.extend(write_cbs)

    default_args = (
        "{\n"
        f"    'owner': {p.owner!r},\n"
        f"    'retries': {p.retries!r},\n"
        f"    'retry_delay': timedelta(minutes={p.retry_delay_minutes!r}),\n"
        "}"
    )
    sd = p.start_date
    dag_lines = [
        "with DAG(",
        f"    {p.dag_id!r},",
        f"    description={p.description!r},",
        f"    schedule={p.schedule!r},",
        f"    start_date=pendulum.datetime({sd.year}, {sd.month}, {sd.day}, tz={p.timezone!r}),",
        "    catchup=False,",
        f"    tags={p.tags!r},",
        f"    default_args={default_args.replace(chr(10), chr(10) + '    ')},",
        ") as dag:",
        "",
    ]
    ind = 4
    first_task = READ_TASK
    if p.wait_for:
        dag_lines.append(
            _call(
                "wait_source",
                "GreenplumTablesWaitSensor",
                {
                    "task_id": "'wait_source'",
                    "tables_wait_list": _lit(p.wait_for.tables, ind + 4),
                    "gp_service": repr(p.gp_service),
                    "timeout_seconds": repr(p.wait_for.timeout_seconds),
                    # Повтор сенсора после таймаута означал бы ещё сутки ожидания.
                    "retries": "0",
                },
                ind,
            )
        )
        dag_lines.append("")
        first_task = "wait_source"

    dag_lines.append(
        _call(
            READ_TASK,
            "GreenplumToDataframeOperator",
            {
                "task_id": repr(READ_TASK),
                "query": _lit(p.source.query, ind + 4),
                "gp_service": repr(p.gp_service),
                "callback": f"{READ_TASK}_callback",
                "mode": repr(p.gp_mode),
                "executor_config": _lit(
                    {
                        "time_limit": p.time_limit,
                        "flavor": p.source.flavor,
                        "output": [{"src": WORK_OUT, "name": "output"}],
                    },
                    ind + 4,
                ),
            },
            ind,
        )
    )
    dag_lines.append("")

    m = p.model
    dag_lines.append(
        _call(
            INFERENCE_TASK,
            "BatchInferenceOperator",
            {
                "task_id": repr(INFERENCE_TASK),
                "predict_py": PREDICTOR_CLASS,
                "mrid": _lit(m.mrid, ind + 4),
                "image": repr(m.image),
                "requirements": _lit(m.requirements, ind + 4),
                "flavor": repr(m.flavor),
                "batch_size": repr(m.batch_size),
                "input_df_path": repr(f"{WORK_IN}/{data}"),
                "output_df_path": repr(f"{WORK_OUT}/{data}"),
                "executor_config": _lit(
                    {
                        "time_limit": p.time_limit,
                        "input": [{"src": f"{READ_TASK}/output", "dst": WORK_IN}],
                        "output": [{"src": WORK_OUT, "name": "output"}],
                    },
                    ind + 4,
                ),
            },
            ind,
        )
    )
    dag_lines.append("")

    groups = []
    for s in p.sinks:
        g = f"write_{s.short_name}"
        groups.append(g)
        dag_lines.append(f'    with TaskGroup("{g}", prefix_group_id=False) as {g}:')
        gi = ind + 4
        chain = [f"{g}_stg", f"{g}_load"]
        dag_lines.append(
            _call(
                f"{g}_stg",
                "DataframeToGreenplumOperator",
                {
                    "task_id": repr(f"{g}_stg"),
                    "callback": f"{g}_callback",
                    "table": repr(s.stg_table),
                    "gp_service": repr(p.gp_service),
                    "if_exists": "'replace'",
                    "columns_types": _lit(s.columns, gi + 4),
                    "mode": repr(p.gp_mode),
                    "executor_config": _lit(
                        {
                            "time_limit": p.time_limit,
                            "flavor": s.flavor,
                            "input": [{"src": f"{INFERENCE_TASK}/output", "dst": WORK_IN}],
                        },
                        gi + 4,
                    ),
                },
                gi,
            )
        )
        dag_lines.append("")
        dag_lines.append(
            _call(
                f"{g}_load",
                "GreenplumExecuteOperator",
                {
                    "task_id": repr(f"{g}_load"),
                    "query": _lit(load_sql(s, p), gi + 4),
                    "gp_service": repr(p.gp_service),
                    "mode": repr(p.gp_mode),
                },
                gi,
            )
        )
        dag_lines.append("")
        for flag, suffix, sql in (
            (s.harmonize, "harmonize", harmonize_sql(s)),
            (s.actualize, "actualize", actualize_sql(s)),
        ):
            if not flag:
                continue
            chain.append(f"{g}_{suffix}")
            dag_lines.append(
                _call(
                    f"{g}_{suffix}",
                    "GreenplumExecuteOperator",
                    {
                        "task_id": repr(f"{g}_{suffix}"),
                        "query": repr(sql),
                        "gp_service": repr(p.gp_service),
                        "mode": repr(p.gp_mode),
                    },
                    gi,
                )
            )
            dag_lines.append("")
        dag_lines.append(" " * gi + " >> ".join(chain))
        dag_lines.append("")

    tail = [first_task, READ_TASK, INFERENCE_TASK] if first_task != READ_TASK else [READ_TASK, INFERENCE_TASK]
    sinks_expr = groups[0] if len(groups) == 1 else "[" + ", ".join(groups) + "]"
    dag_lines.append("    " + " >> ".join(tail + [sinks_expr]))

    blocks.append("\n".join(dag_lines))
    code = "\n\n\n".join(blocks) + "\n"
    compile(code, f"{p.dag_id}.py", "exec")
    return code


def render_dir(pipeline_dir: Union[str, Path], shadow: bool = False) -> tuple[Pipeline, str]:
    pipeline_dir = Path(pipeline_dir)
    pipeline = load_pipeline(pipeline_dir)
    if shadow:
        pipeline = pipeline.as_shadow()
    source = (pipeline_dir / pipeline.model.predictor).read_text(encoding="utf-8")
    name = f"{pipeline_dir.name}/pipeline.yaml"
    return pipeline, render_pipeline(pipeline, source, name)
