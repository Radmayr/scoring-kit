"""Локальный прогон сгенерированного DAG'а: чтение -> инференс -> подготовка записи.

Исполняется ровно тот код, который уйдёт в Airflow (колбэки и класс предиктора
берутся из сгенерированного DAG'а). Вместо Greenplum — локальный файл с выборкой,
вместо Model Registry — локальные пути к моделям. Пути /work/... подменяются на
папки внутри out_dir, по одной на таск, как монтирует executor_config.
"""

from __future__ import annotations

import ast
import contextlib
import os
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from scoring_kit.render import INFERENCE_TASK, PREDICTOR_CLASS, READ_TASK, render_dir
from scoring_kit.stubs import load_dag
from scoring_kit.sql import load_sql


@dataclass
class DebugResult:
    dag_code: str
    scored: pd.DataFrame
    to_write: dict = field(default_factory=dict)  # таблица -> DataFrame, который уйдёт в stg
    sql: dict = field(default_factory=dict)  # таблица -> SQL загрузки из stg


@contextlib.contextmanager
def _work_root(root: Path):
    old = os.environ.get("SCORING_KIT_WORK_ROOT")
    os.environ["SCORING_KIT_WORK_ROOT"] = str(root)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("SCORING_KIT_WORK_ROOT", None)
        else:
            os.environ["SCORING_KIT_WORK_ROOT"] = old


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def operator_predictor(code: str, class_name: str = PREDICTOR_CLASS) -> type:
    """Класс предиктора в том виде, в котором его получает BatchInferenceOperator.

    Оператор берёт inspect.getsource(класса), вырезает подстроку "BasePredictor" и
    переименовывает класс в Predictor; в поде он лежит в predict.py без единого
    имени уровня модуля. Воспроизводим это дословно (см. исходник оператора).
    """
    tree = ast.parse(code)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    source = textwrap.dedent(ast.get_source_segment(code, node))
    source = source.replace("BasePredictor", "").replace(class_name, "Predictor")
    namespace: dict = {}
    exec(compile(source, "predict.py", "exec"), namespace)
    return namespace["Predictor"]


def _local(root: Path, work_path: str) -> Path:
    return Path(str(root) + work_path)


def run_debug(
    pipeline_dir: Union[str, Path],
    data_path: Union[str, Path],
    model_paths: list,
    out_dir: Union[str, Path],
    limit: Optional[int] = None,
) -> DebugResult:
    pipeline, code = render_dir(pipeline_dir)
    dag = load_dag(code, f"{pipeline.dag_id}.py")
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / f"{pipeline.dag_id}.py").write_text(code, encoding="utf-8")
    data = pipeline.data_file

    # 1. read_source: колбэк получает df, как от GreenplumToDataframeOperator.
    df = _read_any(Path(data_path))
    if limit:
        df = df.head(limit)
    read_root = out_dir / READ_TASK
    with _work_root(read_root):
        dag.tasks[READ_TASK].kwargs["callback"](df)
    read_output = _local(read_root, f"/work/output/{data}")

    # 2. inference: как BatchInferenceOperator — setup один раз, predict по батчам.
    op = dag.tasks[INFERENCE_TASK]
    predictor = operator_predictor(code)()
    predictor.setup([str(p) for p in model_paths])
    batch_size = op.kwargs["batch_size"]
    # run.py оператора: pd.read_csv(f, chunksize=batch_size) — типы определяются в каждом чанке отдельно.
    batches = pd.read_csv(read_output, chunksize=batch_size)
    scored = pd.concat([predictor.predict(b) for b in batches], ignore_index=True)
    inf_output = _local(out_dir / INFERENCE_TASK, f"/work/output/{data}")
    inf_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(inf_output, index=False)

    # 3. write_*: колбэк читает выход инференса и готовит df для stg-таблицы.
    result = DebugResult(dag_code=code, scored=scored)
    for sink in pipeline.sinks:
        task_id = f"write_{sink.short_name}_stg"
        root = out_dir / task_id
        dst = _local(root, f"/work/input/{data}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(inf_output, dst)
        with _work_root(root):
            to_write = dag.tasks[task_id].kwargs["callback"]()
        to_write.to_csv(root / "to_write.csv", index=False)
        result.to_write[sink.table] = to_write
        result.sql[sink.table] = load_sql(sink, pipeline)
    return result
