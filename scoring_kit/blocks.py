"""Блоки DAG'а: (блок × движок) → код таска для конкретного оператора платформы.

Блок ничего не знает о рецептах: получает параметры и добавляет в DagBuilder импорты,
колбэки и код таска. Все ограничения операторов (csv-вход, пути /work, самодостаточный
код колбэков) живут здесь.
"""

from __future__ import annotations

from scoring_kit import runtime
from scoring_kit.codegen import IND, DagBuilder, call, func, lit
from scoring_kit.spec import Pipeline, Sink
from scoring_kit.sql import actualize_sql, harmonize_sql, load_sql

WORK_OUT = "/work/output"
WORK_IN = "/work/input"
DATA_FILE = "data.csv"
REPORT_FILE = "report.csv"

GP_OPS = "from airflow_provider_greenplum.operators.greenplum import {}"
GP_SENSOR = "from airflow_provider_greenplum.sensors.greenplum import GreenplumTablesWaitSensor"
INFERENCE_OPS = "from airflow_provider_inference.operators.inference import BasePredictor, BatchInferenceOperator"
PYTHON_OP = "from airflow.operators.python import PythonOperator"
TASK_GROUP = "from airflow.utils.task_group import TaskGroup"
DLH_OPS = "from airflow_provider_dlh.operators.dlh import {}"
DLH_SENSOR = "from airflow_provider_dlh.sensors.dlh import DLHTablesWaitSensor"
DLH_INFERENCE = "from airflow_provider_dlh_inference.operators import DlhBatchInferenceOperator, ModelMeta, OutputType"


def _exec_cfg(p: Pipeline, flavor: str | None = None, inputs=None, output: bool = True) -> dict:
    cfg: dict = {"time_limit": p.time_limit}
    if flavor:
        cfg["flavor"] = flavor
    if inputs:
        cfg["input"] = [{"src": f"{src}/output", "dst": dst} for src, dst in inputs]
    if output:
        cfg["output"] = [{"src": WORK_OUT, "name": "output"}]
    return cfg


# ---------------------------------------------------------------- Greenplum


def gp_sensor(b: DagBuilder, p: Pipeline) -> str:
    b.need(GP_SENSOR)
    b.task(
        call(
            "wait_source",
            "GreenplumTablesWaitSensor",
            {
                "task_id": "'wait_source'",
                "tables_wait_list": lit(p.wait_for.tables, IND + 4),
                "gp_service": repr(p.gp_service),
                "timeout_seconds": repr(p.wait_for.timeout_seconds),
                # Повтор сенсора после таймаута означал бы ещё сутки ожидания.
                "retries": "0",
            },
            IND,
        )
    )
    return "wait_source"


def gp_read(
    b: DagBuilder,
    p: Pipeline,
    task_id: str,
    query: str,
    flavor: str,
    min_rows: int,
    required: list[str],
    unique_key: list[str],
) -> str:
    """Чтение из Greenplum в поде: Decimal → float, проверки входа, csv в /work/output."""
    b.need(GP_OPS.format("GreenplumToDataframeOperator"))
    b.block(
        func(
            f"{task_id}_callback",
            "df",
            (runtime.sk_normalize_decimals, runtime.sk_check_source, runtime.sk_write_frame),
            f"""
            df = sk_normalize_decimals(df)
            sk_check_source(
                df,
                min_rows={min_rows!r},
                required_columns={required!r},
                unique_key={unique_key!r},
            )
            sk_write_frame(df, {f"{WORK_OUT}/{DATA_FILE}"!r})
            """,
        )
    )
    b.task(
        call(
            task_id,
            "GreenplumToDataframeOperator",
            {
                "task_id": repr(task_id),
                "query": lit(query, IND + 4),
                "gp_service": repr(p.gp_service),
                "callback": f"{task_id}_callback",
                "mode": repr(p.gp_mode),
                "executor_config": lit(_exec_cfg(p, flavor), IND + 4),
            },
            IND,
        )
    )
    return task_id


def gp_write(b: DagBuilder, p: Pipeline, sink: Sink, upstream: str) -> str:
    """stg (DataframeToGreenplumOperator) → перенос в транзакции → гранты → актуализация."""
    b.need(
        TASK_GROUP,
        GP_OPS.format("DataframeToGreenplumOperator"),
        GP_OPS.format("GreenplumExecuteOperator"),
    )
    g = f"write_{sink.short_name}"
    file = REPORT_FILE if sink.data == "report" else DATA_FILE
    b.block(
        func(
            f"{g}_callback",
            "",
            (runtime.sk_read_frame, runtime.sk_conform_to_columns),
            f"""
            df = sk_read_frame({f"{WORK_IN}/{file}"!r})
            return sk_conform_to_columns(df, {sink.columns!r})
            """,
        )
    )
    gi = IND + 4
    tasks = [
        call(
            f"{g}_stg",
            "DataframeToGreenplumOperator",
            {
                "task_id": repr(f"{g}_stg"),
                "callback": f"{g}_callback",
                "table": repr(sink.stg_table),
                "gp_service": repr(p.gp_service),
                "if_exists": "'replace'",
                "columns_types": lit(sink.columns, gi + 4),
                "mode": repr(p.gp_mode),
                "executor_config": lit(_exec_cfg(p, sink.flavor, [(upstream, WORK_IN)], output=False), gi + 4),
            },
            gi,
        ),
        call(
            f"{g}_load",
            "GreenplumExecuteOperator",
            {
                "task_id": repr(f"{g}_load"),
                "query": lit(load_sql(sink, p), gi + 4),
                "gp_service": repr(p.gp_service),
                "mode": repr(p.gp_mode),
            },
            gi,
        ),
    ]
    chain = [f"{g}_stg", f"{g}_load"]
    for flag, suffix, sql in (
        (sink.harmonize, "harmonize", harmonize_sql(sink)),
        (sink.actualize, "actualize", actualize_sql(sink)),
    ):
        if not flag:
            continue
        chain.append(f"{g}_{suffix}")
        tasks.append(
            call(
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
    b.group(g, tasks, chain)
    return g


# ---------------------------------------------------------------- ML Core job (BatchInferenceOperator)


def mlc_infer(
    b: DagBuilder,
    p: Pipeline,
    task_id: str,
    class_code: str,
    class_name: str,
    mrids: list[str],
    image: str,
    requirements: list[str],
    flavor: str,
    batch_size: int | None,
    inputs: list[tuple[str, str]],
    input_file: str,
) -> str:
    b.need(INFERENCE_OPS)
    b.block(class_code)
    b.task(
        call(
            task_id,
            "BatchInferenceOperator",
            {
                "task_id": repr(task_id),
                "predict_py": class_name,
                "mrid": lit(mrids, IND + 4),
                "image": repr(image),
                "requirements": lit(requirements, IND + 4),
                "flavor": repr(flavor),
                "batch_size": repr(batch_size),
                "input_df_path": repr(input_file),
                "output_df_path": repr(f"{WORK_OUT}/{DATA_FILE}"),
                "executor_config": lit(_exec_cfg(p, None, inputs), IND + 4),
            },
            IND,
        )
    )
    return task_id


def py_merge(
    b: DagBuilder, p: Pipeline, task_id: str, upstreams: list[str], key: list[str], score_groups: list[list[str]], flavor: str
) -> str:
    """Слияние результатов нескольких job'ов инференса по ключу."""
    b.need(PYTHON_OP)
    inputs = [(u, f"{WORK_IN}_{i}") for i, u in enumerate(upstreams)]
    paths = [f"{dst}/{DATA_FILE}" for _, dst in inputs]
    b.block(
        func(
            f"{task_id}_callback",
            "",
            (runtime.sk_merge_outputs, runtime.sk_write_frame),
            f"""
            df = sk_merge_outputs({paths!r}, {key!r}, {score_groups!r})
            sk_write_frame(df, {f"{WORK_OUT}/{DATA_FILE}"!r})
            """,
        )
    )
    b.task(
        call(
            task_id,
            "PythonOperator",
            {
                "task_id": repr(task_id),
                "python_callable": f"{task_id}_callback",
                "executor_config": lit(_exec_cfg(p, flavor, inputs), IND + 4),
            },
            IND,
        )
    )
    return task_id


# ---------------------------------------------------------------- DLH


def dlh_sensor(b: DagBuilder, p: Pipeline) -> str:
    b.need(DLH_SENSOR)
    b.task(
        call(
            "wait_source",
            "DLHTablesWaitSensor",
            {
                "task_id": "'wait_source'",
                "tables_wait_list": lit(p.wait_for.tables, IND + 4),
                "timeout_seconds": repr(p.wait_for.timeout_seconds),
                "retries": "0",
            },
            IND,
        )
    )
    return "wait_source"


def dlh_prepare(b: DagBuilder, query: str, staging_table: str) -> str:
    """Запрос → таблица в DLH: оператор инференса читает только таблицу целиком."""
    b.need(DLH_OPS.format("DLHExecuteOperator"))
    statements = [f"drop table if exists {staging_table}", f"create table {staging_table} as\n{query}"]
    b.task(
        call(
            "prepare_source",
            "DLHExecuteOperator",
            {"task_id": "'prepare_source'", "query": lit(statements, IND + 4)},
            IND,
        )
    )
    return "prepare_source"


def dlh_check(b: DagBuilder, task_id: str, query: str, helper, call_args: str) -> str:
    b.need(DLH_OPS.format("DLHToDataframeOperator"))
    b.block(func(f"{task_id}_callback", "df", (helper,), f"{helper.__name__}(df, {call_args})"))
    b.task(
        call(
            task_id,
            "DLHToDataframeOperator",
            {"task_id": repr(task_id), "query": lit(query, IND + 4), "callback": f"{task_id}_callback"},
            IND,
        )
    )
    return task_id


def dlh_infer(
    b: DagBuilder,
    task_id: str,
    key_columns: list[str],
    score_column: str,
    mrid: str,
    source_table: str,
    target_table: str,
    image: str,
    max_executors: int | None,
    max_wait: int | None,
) -> str:
    b.need(DLH_INFERENCE)
    meta = (
        "[\n"
        f"{' ' * (IND + 8)}ModelMeta(\n"
        f"{' ' * (IND + 12)}key_columns={lit(key_columns, IND + 12)},\n"
        f"{' ' * (IND + 12)}output_columns={[score_column]!r},\n"
        f"{' ' * (IND + 12)}mrid={mrid!r},\n"
        # без явного типа оператор пишет float32 — теряется точность скора
        f"{' ' * (IND + 12)}output_types={{{score_column!r}: OutputType.DOUBLE}},\n"
        f"{' ' * (IND + 8)}),\n"
        f"{' ' * (IND + 4)}]"
    )
    kwargs = {
        "task_id": repr(task_id),
        "models_meta": meta,
        "source_table": repr(source_table),
        "target_table": repr(target_table),
        "image": repr(image),
        "max_wait": repr(max_wait),
    }
    if max_executors is not None:
        kwargs["max_executors"] = repr(max_executors)
    b.task(call(task_id, "DlhBatchInferenceOperator", kwargs, IND))
    return task_id
