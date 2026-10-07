"""Локальный прогон сгенерированного DAG'а — любого рецепта и движка.

Исполняется ровно тот код, который уйдёт в Airflow: колбэки и классы предикторов берутся из
сгенерированного файла и запускаются так, как это делает платформа:

- GreenplumToDataframeOperator — колбэк получает выборку из локального файла (вместо Greenplum);
- BatchInferenceOperator — класс предиктора в виде, в котором его получает оператор (вырезан
  BasePredictor, класс переименован), setup с локальными путями моделей, predict по батчам
  pd.read_csv(chunksize) как в run.py оператора;
- PythonOperator / DataframeToGreenplumOperator — колбэк с примонтированными входами;
- DLH: подготовка выборки, проверки и инференс эмулируются на pandas, бандл модели
  вызывается так же, как в шаблоне DlhBatchInferenceOperator.

Пути /work/... каждого таска подменяются на out_dir/<task_id>/work/..., входы монтируются
копированием выходов вышестоящих тасков — как executor_config input/output на платформе.
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

from scoring_kit.render import PREDICTOR_CLASS, render_dir
from scoring_kit.sql import load_sql
from scoring_kit.stubs import load_dag

DATA_ALIASES = {"source": "read_source", "dev": "read_dev", "apply": "read_apply"}


@dataclass
class DebugResult:
    dag_code: str
    scored: Optional[pd.DataFrame] = None
    to_write: dict = field(default_factory=dict)  # таблица -> DataFrame для записи
    sql: dict = field(default_factory=dict)  # таблица -> SQL загрузки из stg
    outputs: dict = field(default_factory=dict)  # task_id -> выход таска (DataFrame)


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


def _read_any(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _local(root: Path, work_path: str) -> Path:
    return Path(str(root) + work_path)


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


def _topological(tasks: dict) -> list:
    order, done = [], set()

    def visit(t):
        if t.task_id in done:
            return
        for u in sorted(t.upstream, key=lambda x: x.task_id):
            visit(u)
        done.add(t.task_id)
        order.append(t)

    for t in tasks.values():
        visit(t)
    return order


class _Simulator:
    def __init__(self, pipeline, code, dag, out_dir: Path, data: dict, models: dict, limit: Optional[int]):
        self.p, self.code, self.dag, self.out = pipeline, code, dag, out_dir
        self.data, self.models, self.limit = data, models, limit
        self.result = DebugResult(dag_code=code)
        self.tables: dict[str, pd.DataFrame] = {}  # эмуляция таблиц DLH

    def root(self, task_id: str) -> Path:
        return self.out / task_id

    def mount(self, task) -> None:
        for item in (task.kwargs.get("executor_config") or {}).get("input", []):
            src_task = item["src"].split("/")[0]
            src = _local(self.root(src_task), "/work/output")
            dst = _local(self.root(task.task_id), item["dst"])
            dst.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)

    def output(self, task_id: str, name: str = "data.csv") -> Path:
        return _local(self.root(task_id), f"/work/output/{name}")

    def input_frame(self, task_id: str) -> pd.DataFrame:
        key = task_id if task_id in self.data else next((k for k, v in DATA_ALIASES.items() if v == task_id), None)
        path = self.data.get(task_id) or (self.data.get(key) if key else None) or self.data.get("*")
        if path is None:
            raise ValueError(f"нет данных для {task_id}: передайте --data {task_id}=файл")
        df = _read_any(path)
        return df.head(self.limit) if self.limit else df

    # ---------------------------------------------------------------- операторы

    def GreenplumToDataframeOperator(self, t):
        with _work_root(self.root(t.task_id)):
            t.kwargs["callback"](self.input_frame(t.task_id))

    def BatchInferenceOperator(self, t):
        self.mount(t)
        k = t.kwargs
        cls = operator_predictor(self.code, k["predict_py"].__name__)
        paths = []
        for mrid in k["mrid"]:
            if mrid not in self.models:
                raise ValueError(f"нет файла модели для {mrid}: передайте --model {mrid}=файл")
            paths.append(str(self.models[mrid]))
        root = self.root(t.task_id)
        inp = _local(root, k["input_df_path"])
        out = _local(root, k["output_df_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with _work_root(root):
            predictor = cls()
            predictor.setup(paths if paths else [""])
            # run.py оператора: pd.read_csv(f, chunksize=batch_size) или целиком, если batch_size=None
            if k["batch_size"]:
                parts = [predictor.predict(chunk) for chunk in pd.read_csv(inp, chunksize=k["batch_size"])]
                scored = pd.concat(parts, ignore_index=True)
            else:
                scored = predictor.predict(pd.read_csv(inp))
        scored.to_csv(out, index=False)
        # В памяти, до круга через csv: так сверка с эталоном проверяет модель, а не парсер чисел.
        self.result.outputs[t.task_id] = scored

    def PythonOperator(self, t):
        self.mount(t)
        with _work_root(self.root(t.task_id)):
            t.kwargs["python_callable"]()

    def DataframeToGreenplumOperator(self, t):
        self.mount(t)
        with _work_root(self.root(t.task_id)):
            df = t.kwargs["callback"]()
        df.to_csv(self.root(t.task_id) / "to_write.csv", index=False)
        table = t.kwargs["table"]
        self.result.to_write[table[: -len("_stg")] if table.endswith("_stg") else table] = df

    def GreenplumExecuteOperator(self, t):
        pass

    def GreenplumTablesWaitSensor(self, t):
        pass

    DLHTablesWaitSensor = GreenplumTablesWaitSensor

    def DLHExecuteOperator(self, t):
        # prepare_source: результат запроса = переданный файл выборки
        for stmt in t.kwargs["query"]:
            if stmt.startswith("create table "):
                table = stmt.split()[2]
                self.tables[table] = self.input_frame("read_source")

    def _source_table(self) -> pd.DataFrame:
        name = self.p.dlh.staging_table if self.p.source.query else self.p.source.table
        if name not in self.tables:
            self.tables[name] = self.input_frame("read_source")
        return self.tables[name]

    def DLHToDataframeOperator(self, t):
        src = self._source_table()
        if t.task_id == "check_source":
            missing = [c for c in _required(self.p) if c not in src.columns]
            if missing:
                raise ValueError(f"Trino: column(s) {missing} cannot be resolved")
            key = self.p.source.checks.unique_key
            dup = int(src.groupby(key, dropna=False).size().gt(1).sum()) if key else 0
            stats = pd.DataFrame([{"n_rows": len(src), "dup_keys": dup, "columns_ok": 1}])
        else:
            res = self.tables[self.p.sinks[0].table]
            score = res[self.p.model.score_column]
            stats = pd.DataFrame([{
                "n_rows": len(res), "n_score": int(score.notna().sum()), "min_score": score.min(),
                "max_score": score.max(), "n_source": len(src),
            }])
        t.kwargs["callback"](stats)

    def DlhBatchInferenceOperator(self, t):
        import pickle

        from scoring_kit.bundle import bundle_from_file

        meta = t.kwargs["models_meta"][0]
        path = self.models.get(meta.mrid)
        if path is None:
            raise ValueError(f"нет файла модели для {meta.mrid}: передайте --model {meta.mrid}=файл")
        try:
            model = pickle.loads(Path(path).read_bytes())
            if not hasattr(model, "features"):
                raise TypeError
        except Exception:
            # сырой файл модели: упаковываем на лету так же, как `scoring bundle`
            model = pickle.loads(pickle.dumps(bundle_from_file(path, self.p.model)))
        src = self._source_table()
        # Шаблон оператора: predict получает только колонки model.features, результат
        # раскладывается по output_columns; key_columns переносятся из входа как есть.
        batch = pd.DataFrame({c: src[c] for c in model.features})
        preds = model.predict(batch)
        result = src[list(meta.key_columns)].copy()
        for col in meta.output_columns:
            result[col] = preds[col].to_numpy()
        result["processed_dttm"] = pd.Timestamp.now()
        self.tables[t.kwargs["target_table"]] = result
        self.root(t.task_id).mkdir(parents=True, exist_ok=True)
        result.to_csv(self.root(t.task_id) / "target.csv", index=False)
        self.result.to_write[t.kwargs["target_table"]] = result

    # ---------------------------------------------------------------- прогон

    def run(self) -> DebugResult:
        for t in _topological(self.dag.tasks):
            handler = getattr(self, type(t).__name__, None)
            if handler is None:
                raise NotImplementedError(f"нет эмуляции оператора {type(t).__name__}")
            handler(t)
            out = self.output(t.task_id)
            if out.exists() and t.task_id not in self.result.outputs:
                self.result.outputs[t.task_id] = pd.read_csv(out, low_memory=False)
        p = self.p
        if p.engine == "dlh":
            self.result.scored = self.tables.get(p.sinks[0].table)
        else:
            producer = next(
                (u.task_id for t in self.dag.tasks.values() if type(t).__name__ == "DataframeToGreenplumOperator"
                 for u in t.upstream),
                None,
            )
            self.result.scored = self.result.outputs.get(producer)
            for sink in p.sinks:
                self.result.sql[sink.table] = load_sql(sink, p)
        return self.result


def _required(p) -> list[str]:
    from scoring_kit.recipes import required_source_columns

    return required_source_columns(p)


def _mrid_map(pipeline, model_paths) -> dict:
    """--model: пути по порядку mrid в yaml, либо mrid=путь / имя_модели=путь."""
    if isinstance(model_paths, dict):
        named = dict(model_paths)
    else:
        named, positional = {}, []
        for item in model_paths or []:
            item = str(item)
            if "=" in item and not Path(item).exists():
                k, v = item.split("=", 1)
                named[k] = v
            else:
                positional.append(item)
        mrids = [mrid for m in pipeline.all_models for mrid in m.mrid]
        if len(positional) > len(mrids):
            raise ValueError(f"моделей передано {len(positional)}, а в yaml {len(mrids)}")
        named.update(dict(zip(mrids, positional)))
    by_name = {m.name: m.mrid[0] for m in pipeline.all_models if m.name}
    return {by_name.get(k, k): v for k, v in named.items()}


def run_debug(
    pipeline_dir: Union[str, Path],
    data_path: Union[str, Path, dict],
    model_paths,
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
    data = data_path if isinstance(data_path, dict) else {"*": data_path}
    sim = _Simulator(pipeline, code, dag, out_dir, data, _mrid_map(pipeline, model_paths), limit)
    return sim.run()
