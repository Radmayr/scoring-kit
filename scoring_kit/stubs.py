"""Заглушки airflow и airflow_provider_* для локального исполнения сгенерированного DAG.

Операторы ничего не запускают — только запоминают аргументы и связи между
тасками. Этого хватает, чтобы проверить граф и достать из DAG'а колбэки и класс
предиктора для локальной отладки ровно того кода, который уйдёт в Airflow.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import sys
import types

_dag_stack: list = []
_group_stack: list = []


class StubDAG:
    def __init__(self, dag_id, **kwargs):
        self.dag_id = dag_id
        self.kwargs = kwargs
        self.tasks: dict = {}

    def __enter__(self):
        _dag_stack.append(self)
        return self

    def __exit__(self, *exc):
        _dag_stack.pop()


def _tasks_of(node) -> list:
    return node.tasks if isinstance(node, StubTaskGroup) else [node]


def _link(left, right):
    for lnode in left if isinstance(left, list) else [left]:
        for rnode in right if isinstance(right, list) else [right]:
            ups = _tasks_of(lnode)
            downs = _tasks_of(rnode)
            # Для группы: входы — таски без предков внутри группы, выходы — без потомков.
            if isinstance(lnode, StubTaskGroup):
                ups = [t for t in ups if not any(t in d.upstream for d in lnode.tasks)]
            if isinstance(rnode, StubTaskGroup):
                downs = [t for t in downs if not any(u in t.upstream for u in rnode.tasks)]
            for d in downs:
                d.upstream.update(ups)


class _Node:
    def __rshift__(self, other):
        _link(self, other)
        return other

    def __rrshift__(self, other):
        _link(other, self)
        return self

    def __lshift__(self, other):
        _link(other, self)
        return other


class StubOperator(_Node):
    def __init__(self, task_id, **kwargs):
        self.task_id = task_id
        self.kwargs = kwargs
        self.upstream: set = set()
        if not _dag_stack:
            raise RuntimeError(f"оператор {task_id} создан вне DAG")
        dag = _dag_stack[-1]
        if task_id in dag.tasks:
            raise ValueError(f"task_id {task_id} уже есть в DAG")
        dag.tasks[task_id] = self
        if _group_stack:
            _group_stack[-1].tasks.append(self)

    def __repr__(self):
        return f"<{type(self).__name__} {self.task_id}>"


class StubTaskGroup(_Node):
    def __init__(self, group_id, prefix_group_id=True, **kwargs):
        self.group_id = group_id
        self.tasks: list = []

    def __enter__(self):
        _group_stack.append(self)
        return self

    def __exit__(self, *exc):
        _group_stack.pop()


def _op(name: str):
    return type(name, (StubOperator,), {})


class BasePredictor:
    def setup(self, model_paths):
        pass

    def predict(self, df):
        raise NotImplementedError


class ModelMeta:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"ModelMeta({self.__dict__})"


class OutputType:
    def __init__(self, type_name, args=()):
        self.type_name = type_name
        self.args = args

    def __eq__(self, other):
        return isinstance(other, OutputType) and (self.type_name, self.args) == (other.type_name, other.args)

    def __repr__(self):
        return f"OutputType({self.type_name!r})"


for _name, _type in (
    ("FLOAT", "FloatType"), ("DOUBLE", "DoubleType"), ("INT", "IntegerType"), ("LONG", "LongType"),
    ("STRING", "StringType"), ("DATE", "DateType"), ("TIMESTAMP", "TimestampType"), ("BOOL", "BooleanType"),
):
    setattr(OutputType, _name, OutputType(_type))


class TiMeNotifier:
    def __init__(self, message, recipients, **kwargs):
        self.message = message
        self.recipients = recipients

    def __call__(self, context):
        print(f"[stub TiMe] {self.recipients}: {self.message}")


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _build_modules() -> dict:
    gp_ops = _module(
        "airflow_provider_greenplum.operators.greenplum",
        GreenplumToDataframeOperator=_op("GreenplumToDataframeOperator"),
        DataframeToGreenplumOperator=_op("DataframeToGreenplumOperator"),
        GreenplumExecuteOperator=_op("GreenplumExecuteOperator"),
    )
    gp_sensors = _module(
        "airflow_provider_greenplum.sensors.greenplum",
        GreenplumTablesWaitSensor=_op("GreenplumTablesWaitSensor"),
    )
    inference = _module(
        "airflow_provider_inference.operators.inference",
        BasePredictor=BasePredictor,
        BatchInferenceOperator=_op("BatchInferenceOperator"),
    )
    python_ops = _module("airflow.operators.python", PythonOperator=_op("PythonOperator"))
    task_group = _module("airflow.utils.task_group", TaskGroup=StubTaskGroup)

    def _datetime(year, month, day, *args, tz=None, **kwargs):
        return dt.datetime(year, month, day, *args)

    dlh_ops = _module(
        "airflow_provider_dlh.operators.dlh",
        DLHExecuteOperator=_op("DLHExecuteOperator"),
        DLHToDataframeOperator=_op("DLHToDataframeOperator"),
    )
    dlh_sensors = _module("airflow_provider_dlh.sensors.dlh", DLHTablesWaitSensor=_op("DLHTablesWaitSensor"))
    dlh_inference = _module(
        "airflow_provider_dlh_inference.operators",
        DlhBatchInferenceOperator=_op("DlhBatchInferenceOperator"),
        ModelMeta=ModelMeta,
        OutputType=OutputType,
    )
    time_notifications = _module("airflow_provider_time.notifications", TiMeNotifier=TiMeNotifier)

    return {
        "airflow_provider_dlh": _module("airflow_provider_dlh"),
        "airflow_provider_dlh.operators": _module("airflow_provider_dlh.operators"),
        "airflow_provider_dlh.operators.dlh": dlh_ops,
        "airflow_provider_dlh.sensors": _module("airflow_provider_dlh.sensors"),
        "airflow_provider_dlh.sensors.dlh": dlh_sensors,
        "airflow_provider_dlh_inference": _module("airflow_provider_dlh_inference"),
        "airflow_provider_dlh_inference.operators": dlh_inference,
        "airflow_provider_time": _module("airflow_provider_time"),
        "airflow_provider_time.notifications": time_notifications,
        "airflow": _module("airflow", DAG=StubDAG),
        "airflow.operators": _module("airflow.operators"),
        "airflow.operators.python": python_ops,
        "airflow.utils": _module("airflow.utils"),
        "airflow.utils.task_group": task_group,
        "airflow_provider_greenplum": _module("airflow_provider_greenplum"),
        "airflow_provider_greenplum.operators": _module("airflow_provider_greenplum.operators"),
        "airflow_provider_greenplum.operators.greenplum": gp_ops,
        "airflow_provider_greenplum.sensors": _module("airflow_provider_greenplum.sensors"),
        "airflow_provider_greenplum.sensors.greenplum": gp_sensors,
        "airflow_provider_inference": _module("airflow_provider_inference"),
        "airflow_provider_inference.operators": _module("airflow_provider_inference.operators"),
        "airflow_provider_inference.operators.inference": inference,
        "pendulum": _module("pendulum", datetime=_datetime),
    }


@contextlib.contextmanager
def installed():
    """Временно подменяет модули airflow/провайдеров/pendulum заглушками."""
    fake = _build_modules()
    saved = {name: sys.modules.get(name) for name in fake}
    sys.modules.update(fake)
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def load_dag(code: str, filename: str = "<dag>") -> StubDAG:
    """Исполняет код DAG'а с заглушками и возвращает объект DAG."""
    with installed():
        namespace: dict = {"__name__": "scoring_kit_dag"}
        exec(compile(code, filename, "exec"), namespace)
    dags = [v for v in namespace.values() if isinstance(v, StubDAG)]
    if len(dags) != 1:
        raise RuntimeError(f"в коде ожидался один DAG, найдено {len(dags)}")
    return dags[0]
