# DAG-разведчик: печатает в лог версии библиотек и исходники провайдеров
# airflow_provider_* прямо из среды Airflow (в ML Core этих библиотек нет).
# Опубликовать на тестовый инстанс, запустить вручную (Trigger DAG), прислать логи
# обоих тасков. Ничего не пишет и не меняет. После разведки DAG можно удалить.
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow_provider_greenplum.operators.greenplum import GreenplumToDataframeOperator


def introspect_callback():
    import importlib
    import inspect
    import pkgutil
    import platform
    import sys
    from pathlib import Path

    print("=== python", sys.version, platform.platform())
    for name in ("airflow", "pandas", "numpy", "pyarrow", "fastparquet", "psycopg2", "sqlalchemy", "cloudpickle", "dill"):
        try:
            mod = importlib.import_module(name)
            print(f"=== {name} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"=== {name} НЕТ: {e!r}")

    for pkg_name in ("airflow_provider_inference", "airflow_provider_greenplum", "airflow_provider_dlh"):
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as e:
            print(f"=== {pkg_name} НЕ ИМПОРТИРУЕТСЯ: {e!r}")
            continue
        root = Path(pkg.__file__).parent
        print(f"\n########## {pkg_name} {getattr(pkg, '__version__', '')} {root}")
        for info in pkgutil.walk_packages([str(root)], prefix=pkg_name + "."):
            print("   модуль:", info.name)
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            print(f"\n----- FILE {path.relative_to(root.parent)} ({len(text)} символов)")
            print(text[:60000])

    from airflow_provider_greenplum.operators import greenplum as gp_ops
    from airflow_provider_inference.operators import inference as inf_ops

    for op in (
        inf_ops.BatchInferenceOperator,
        gp_ops.GreenplumToDataframeOperator,
        gp_ops.DataframeToGreenplumOperator,
        gp_ops.GreenplumExecuteOperator,
    ):
        print(f"\n=== {op.__name__}{inspect.signature(op.__init__)}")
        print("template_fields:", getattr(op, "template_fields", None))


def gp_version_callback(df):
    print("=== Greenplum:")
    print(df.to_string())


with DAG(
    "scoring_kit_introspect",
    description="Разведка окружения для scoring-kit (только чтение)",
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["scoring-kit"],
) as dag:

    introspect = PythonOperator(
        task_id="introspect",
        python_callable=introspect_callback,
        executor_config={"time_limit": "7d", "flavor": "2cpu-4ram"},
    )

    gp_version = GreenplumToDataframeOperator(
        task_id="gp_version",
        query="select version()",
        gp_service="vrcl",
        callback=gp_version_callback,
        mode="dal",
        executor_config={"time_limit": "7d", "flavor": "2cpu-4ram"},
    )
