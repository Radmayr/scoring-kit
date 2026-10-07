# DAG-разведчик: печатает в лог версии библиотек, каталог операторов и исходники ключевых
# провайдеров прямо из среды Airflow (в ML Core этих библиотек нет), плюс два эксперимента.
# Ничего не пишет в хранилища. Запуск только вручную (Trigger DAG), по расписанию не стартует.
#
# ВНИМАНИЕ: mlc airflow publish заменяет всё, что вы публиковали через mlc. Кладите этот файл
# к остальным своим DAG'ам и публикуйте папку целиком, предварительно проверив --check.
#
# Перед публикацией подставьте образ инференса из любого своего DAG'а (параметр image=...):
IMAGE = "ВСТАВЬТЕ_ОБРАЗ_ИЗ_СВОЕГО_DAG"

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow_provider_greenplum.operators.greenplum import GreenplumToDataframeOperator
from airflow_provider_inference.operators.inference import BasePredictor, BatchInferenceOperator


def introspect_callback():
    import importlib
    import platform
    import sys

    print("=== python", sys.version, platform.platform())
    for name in (
        "airflow", "pandas", "numpy", "pyarrow", "psycopg2", "sqlalchemy", "cloudpickle", "dill",
        "sklearn", "lightgbm", "catboost", "xgboost", "scipy",
    ):
        try:
            mod = importlib.import_module(name)
            print(f"=== {name} {getattr(mod, '__version__', '?')}")
        except BaseException as e:
            print(f"=== {name} НЕТ: {e!r}")


def operators_catalog_callback():
    """Сначала исходники ключевых провайдеров (уведомления, DLH-инференс), затем каталог
    операторов всех провайдеров. Импорты защищены от sys.exit() внутри модулей."""
    import importlib
    import importlib.metadata as md
    import inspect
    import pkgutil
    import sys
    from pathlib import Path

    def safe_import(name):
        try:
            return importlib.import_module(name), None
        except BaseException as e:  # SystemExit тоже: некоторые модули — исполняемые скрипты
            return None, e

    def dump_sources(pkg_name):
        pkg, err = safe_import(pkg_name)
        if pkg is None:
            print(f"\n=== {pkg_name} НЕ ИМПОРТИРУЕТСЯ: {err!r}", flush=True)
            return
        root = Path(pkg.__file__).parent
        print(f"\n########## ИСХОДНИК {pkg_name} {root}", flush=True)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in (".py", ".j2", ".sql", ".yaml", ".json"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            print(f"\n----- FILE {path.relative_to(root.parent)} ({len(text)} символов)")
            print(text[:60000], flush=True)

    print("=== Пакеты провайдеров и платформы", flush=True)
    for dist in sorted(md.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
        name = (dist.metadata["Name"] or "").lower()
        if any(k in name for k in ("airflow-provider", "mlcore", "ml-core", "dal", "dlh", "notif")):
            print(f"  {dist.metadata['Name']}=={dist.version}")

    # 1. Самое нужное — первым, чтобы не потерялось при любом сбое ниже.
    for pkg_name in ("airflow_provider_time", "tnotifier", "airflow_provider_dlh_inference"):
        dump_sources(pkg_name)

    # 2. Каталог операторов/нотификаторов остальных провайдеров.
    from airflow.models import BaseOperator

    try:
        from airflow.notifications.basenotifier import BaseNotifier
    except BaseException:
        BaseNotifier = None

    top = sorted(m.name for m in pkgutil.iter_modules() if m.name.startswith(("airflow_provider_", "mlcore")))
    print("\n=== Модули верхнего уровня:", top, flush=True)
    for pkg_name in top:
        pkg, err = safe_import(pkg_name)
        if pkg is None:
            print(f"\n##### {pkg_name}: не импортируется {err!r}")
            continue
        print(f"\n##### {pkg_name}", flush=True)
        paths = getattr(pkg, "__path__", None)
        names = [pkg_name] + ([m.name for m in pkgutil.walk_packages(paths, pkg_name + ".", onerror=lambda n: None)] if paths else [])
        for mod_name in names:
            if mod_name.rsplit(".", 1)[-1] in ("__main__", "main") or ".jobs" in mod_name:
                print(f"  {mod_name}: пропущен (исполняемый модуль)")
                continue
            mod, err = safe_import(mod_name)
            if mod is None:
                print(f"  {mod_name}: не импортируется {err!r}")
                continue
            for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod_name:
                    continue
                is_op = issubclass(cls, BaseOperator)
                is_notifier = BaseNotifier is not None and issubclass(cls, BaseNotifier)
                if not (is_op or is_notifier):
                    continue
                try:
                    sig = str(inspect.signature(cls.__init__))
                except BaseException:
                    sig = "(?)"
                doc = (inspect.getdoc(cls) or "").strip().splitlines()
                print(f"  [{'NOTIFIER' if is_notifier else 'OPERATOR'}] {mod_name}.{cls_name}{sig}")
                if doc:
                    print(f"      {doc[0]}")
            for fn_name, fn in inspect.getmembers(mod, inspect.isfunction):
                if fn.__module__ == mod_name and any(k in fn_name.lower() for k in ("notif", "alert", "callback", "send")):
                    print(f"  [FUNC] {mod_name}.{fn_name}{inspect.signature(fn)}")
        sys.stdout.flush()
    print("\n=== КАТАЛОГ ЗАВЕРШЁН", flush=True)


def gp_version_callback(df):
    print("=== Greenplum:")
    print(df.to_string())


# --- Эксперимент: можно ли запустить BatchInferenceOperator без модели (mrid=[]) ---
# Нужен для обучения «на лету» (калибровка): сейчас для этого передают чужую модель.
def make_tiny_input():
    from pathlib import Path

    import pandas as pd

    out = Path("/work/output")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [0.1, 0.5, 0.9], "y": [0, 1, 1]}).to_csv(out / "data.csv", index=False)


class no_model_predictor(BasePredictor):
    def setup(self, model_paths: list[str]):
        print("model_paths:", model_paths)

    def predict(self, df):
        import importlib

        for name in ("sklearn", "lightgbm", "catboost", "scipy"):
            try:
                print(name, importlib.import_module(name).__version__)
            except BaseException as e:
                print(name, "НЕТ", repr(e))
        df["ok"] = 1
        return df


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

    tiny_input = PythonOperator(
        task_id="tiny_input",
        python_callable=make_tiny_input,
        executor_config={"time_limit": "7d", "flavor": "2cpu-4ram", "output": [{"src": "/work/output", "name": "output"}]},
    )

    no_model_inference = BatchInferenceOperator(
        task_id="no_model_inference",
        predict_py=no_model_predictor,
        mrid=[],
        image=IMAGE,
        requirements=["scikit-learn"],
        flavor="2cpu-4ram",
        batch_size=None,
        input_df_path="/work/input/data.csv",
        output_df_path="/work/output/data.csv",
        executor_config={
            "time_limit": "7d",
            "input": [{"src": "tiny_input/output", "dst": "/work/input"}],
            "output": [{"src": "/work/output", "name": "output"}],
        },
    )

    tiny_input >> no_model_inference
