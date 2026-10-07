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


def operators_catalog_callback():
    """Каталог всего, что установлено на инстансе: пакеты провайдеров, операторы, сенсоры,
    нотификаторы и колбэки уведомлений. Плюс полный исходник провайдера DLH-инференса."""
    import importlib
    import importlib.metadata as md
    import inspect
    import pkgutil
    from pathlib import Path

    print("=== Пакеты провайдеров и платформы")
    for dist in sorted(md.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
        name = (dist.metadata["Name"] or "").lower()
        if any(k in name for k in ("airflow-provider", "apache-airflow-providers", "mlcore", "ml-core", "dal", "dlh", "notif")):
            print(f"  {dist.metadata['Name']}=={dist.version}")

    from airflow.models import BaseOperator

    try:
        from airflow.notifications.basenotifier import BaseNotifier
    except Exception:
        BaseNotifier = None

    top = sorted(
        m.name for m in pkgutil.iter_modules()
        if m.name.startswith(("airflow_provider_", "mlcore")) or "notif" in m.name
    )
    print("\n=== Модули верхнего уровня:", top)
    for pkg_name in top:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as e:
            print(f"\n##### {pkg_name}: не импортируется {e!r}")
            continue
        print(f"\n##### {pkg_name}")
        paths = getattr(pkg, "__path__", None)
        modules = [pkg_name] + ([m.name for m in pkgutil.walk_packages(paths, pkg_name + ".")] if paths else [])
        for mod_name in modules:
            try:
                mod = importlib.import_module(mod_name)
            except Exception as e:
                print(f"  {mod_name}: не импортируется {e!r}")
                continue
            for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod_name:
                    continue
                is_op = issubclass(cls, BaseOperator)
                is_notifier = BaseNotifier is not None and issubclass(cls, BaseNotifier)
                if not (is_op or is_notifier):
                    continue
                kind = "NOTIFIER" if is_notifier else "OPERATOR"
                try:
                    sig = str(inspect.signature(cls.__init__))
                except Exception:
                    sig = "(?)"
                doc = (inspect.getdoc(cls) or "").strip().splitlines()
                print(f"  [{kind}] {mod_name}.{cls_name}{sig}")
                if doc:
                    print(f"      {doc[0]}")
            for fn_name, fn in inspect.getmembers(mod, inspect.isfunction):
                if fn.__module__ == mod_name and any(k in fn_name.lower() for k in ("notif", "alert", "callback", "send")):
                    print(f"  [FUNC] {mod_name}.{fn_name}{inspect.signature(fn)}")

    for pkg_name in ("airflow_provider_dlh_inference",):
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as e:
            print(f"\n=== {pkg_name} НЕ ИМПОРТИРУЕТСЯ: {e!r}")
            continue
        root = Path(pkg.__file__).parent
        print(f"\n########## ИСХОДНИК {pkg_name} {root}")
        for path in sorted(root.rglob("*")):
            if path.suffix not in (".py", ".j2", ".sql") and not path.name.endswith(".j2.py"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            print(f"\n----- FILE {path.relative_to(root.parent)} ({len(text)} символов)")
            print(text[:60000])


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

    operators_catalog = PythonOperator(
        task_id="operators_catalog",
        python_callable=operators_catalog_callback,
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
