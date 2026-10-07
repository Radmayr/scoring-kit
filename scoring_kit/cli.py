"""scoring — командная строка scoring-kit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scoring_kit.spec import PipelineError, load_pipeline

PIPELINE_TEMPLATE = """\
dag_id: {name}
description: ""
owner: your.login
schedule: "30 4 * * *"      # cron, время в timezone
timezone: UTC
tags: []
gp_service: vrcl

wait_for:
  tables: [schema.source_table]

source:
  query: |
    select *
    from schema.source_table
  checks:
    min_rows: 1
    unique_key: [id]

model:
  mrid: [tenant/model_name/0.0.1]
  image: registry/path/image:tag
  requirements: []
  features: [feature_1, feature_2]
  cat_features: []
  num_dtype: float64
  cat_as: category

sinks:
  - table: schema.{name}_scores
    mode: replace
    columns:
      id: bigint
      score: numeric
"""

PREDICTOR_TEMPLATE = '''\
import joblib

from scoring_kit import BasePredictor  # в DAG заменится на airflow_provider_inference


class Predictor(BasePredictor):
    def setup(self, model_paths: list[str]):
        # model_paths — пути к артефактам из model.mrid в том же порядке
        self.model = joblib.load(model_paths[0])

    def predict(self, df):
        # Фичи в df уже приведены к типам контракта; порядок — self.features.
        df["score"] = self.model.predict_proba(df[self.features])[:, 1]
        return df
'''


def _cmd_new(args) -> int:
    target = Path(args.dir)
    if target.exists() and any(target.iterdir()):
        print(f"{target} уже существует и не пуст", file=sys.stderr)
        return 1
    target.mkdir(parents=True, exist_ok=True)
    (target / "pipeline.yaml").write_text(PIPELINE_TEMPLATE.format(name=target.name), encoding="utf-8")
    (target / "predictor.py").write_text(PREDICTOR_TEMPLATE, encoding="utf-8")
    print(f"Создан {target}: pipeline.yaml, predictor.py")
    return 0


def _cmd_validate(args) -> int:
    from scoring_kit.render import render_dir

    for d in args.dirs:
        pipeline, _ = render_dir(d)
        print(f"OK {d}: dag_id={pipeline.dag_id}, sinks={[s.table for s in pipeline.sinks]}")
    return 0


def _cmd_render(args) -> int:
    from scoring_kit.render import render_dir

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in args.dirs:
        pipeline, code = render_dir(d, shadow=args.shadow)
        path = out / f"{pipeline.dag_id}.py"
        path.write_text(code, encoding="utf-8")
        print(f"{d} -> {path}")
    return 0


def _parse_data(items: list[str]):
    """--data файл | --data имя=файл (имена: source, dev, apply или task_id чтения)."""
    named = {}
    for item in items:
        if "=" in item and not Path(item).exists():
            k, v = item.split("=", 1)
            named[k] = v
        else:
            named["*"] = item
    return named


def _cmd_debug(args) -> int:
    import pandas as pd

    from scoring_kit.debug import run_debug

    result = run_debug(args.dir, _parse_data(args.data), args.model or [], args.out, limit=args.limit)
    pipeline = load_pipeline(args.dir)
    scored = result.scored
    print(f"\nРезультат: {len(scored)} строк")
    cols = [c for c in pipeline.output_columns if c in scored.columns]
    print(scored[cols].describe().to_string())
    with pd.option_context("display.max_columns", None, "display.width", 200):
        for table, df in result.to_write.items():
            print(f"\n--- {table}: {len(df)} строк к записи ---")
            print(df.head())
            if table in result.sql:
                print(f"\nSQL загрузки из stg:\n{result.sql[table]}")
    print(f"\nФайлы шагов и сгенерированный DAG: {Path(args.out).resolve()}")
    return 0


def _cmd_bundle(args) -> int:
    import pickle

    from scoring_kit.bundle import bundle_from_file

    pipeline = load_pipeline(args.dir)
    if pipeline.model is None:
        print("Ошибка: scoring bundle — для recipe: single_model", file=sys.stderr)
        return 1
    bundle = bundle_from_file(args.model, pipeline.model)
    out = Path(args.out)
    out.write_bytes(pickle.dumps(bundle))
    # Самопроверка: бандл восстанавливается из файла.
    restored = pickle.loads(out.read_bytes())
    print(f"Бандл: {out} ({out.stat().st_size // 1024} КБ), фичей: {len(restored.features)}")
    print("Загрузите файл в Model Registry новой версией и укажите её mrid в model.mrid.")
    return 0


def _cmd_catalog(args) -> int:
    from scoring_kit.catalog import build_catalog

    text = build_catalog([Path(d) for d in args.dirs])
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Каталог: {args.out}")
    else:
        print(text)
    return 0


def _cmd_ddl(args) -> int:
    from scoring_kit.sql import ddl

    pipeline = load_pipeline(args.dir)
    if args.shadow:
        pipeline = pipeline.as_shadow()
    for s in pipeline.sinks:
        print(ddl(s, pipeline) + "\n")
    return 0


def _cmd_compare_sql(args) -> int:
    from scoring_kit.sql import compare_sql

    pipeline = load_pipeline(args.dir)
    shadow = pipeline.as_shadow()
    default_key = pipeline.apply.key if pipeline.recipe == "fit_apply" else pipeline.source.checks.unique_key
    key = args.key.split(",") if args.key else default_key
    if not key:
        print("Укажите --key или source.checks.unique_key", file=sys.stderr)
        return 1
    for prod, sh in zip(pipeline.sinks, shadow.sinks):
        if prod.data != "result":
            continue
        for column in [c for c in pipeline.output_columns if c in prod.columns]:
            print(f"-- {prod.table}: {column}")
            print(compare_sql(prod.table, sh.table, key, column, args.tol) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scoring", description="yaml + predictor.py -> Airflow DAG скоринга")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("new", help="создать заготовку пайплайна")
    p.add_argument("dir")
    p.set_defaults(func=_cmd_new)

    p = sub.add_parser("validate", help="проверить конфиг и предиктор")
    p.add_argument("dirs", nargs="+")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("render", help="сгенерировать .py DAG")
    p.add_argument("dirs", nargs="+")
    p.add_argument("-o", "--out", default="build/dags")
    p.add_argument("--shadow", action="store_true", help="dag_id и таблицы с суффиксом _shadow")
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser("debug", help="локально прогнать DAG: чтение -> инференс -> подготовку записи")
    p.add_argument("dir")
    p.add_argument("--data", required=True, action="append",
                   help="выборка csv вместо запроса; для fit_apply: --data dev=файл --data apply=файл")
    p.add_argument("--model", action="append",
                   help="файл модели; по порядку mrid в yaml или имя_модели=файл; для fit_apply не нужен")
    p.add_argument("--out", default="debug_out")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=_cmd_debug)

    p = sub.add_parser("bundle", help="упаковать модель для engine: dlh (переносимый бандл с контрактом фичей)")
    p.add_argument("dir")
    p.add_argument("--model", required=True, help="файл исходной модели (joblib/pickle LightGBM или CatBoost)")
    p.add_argument("-o", "--out", required=True, help="куда сохранить бандл (.pkl)")
    p.set_defaults(func=_cmd_bundle)

    p = sub.add_parser("catalog", help="каталог процессов: владельцы, модели, таблицы, зависимости")
    p.add_argument("dirs", nargs="+")
    p.add_argument("-o", "--out", help="файл (например CATALOG.md); по умолчанию — в консоль")
    p.set_defaults(func=_cmd_catalog)

    p = sub.add_parser("ddl", help="create table для приёмников")
    p.add_argument("dir")
    p.add_argument("--shadow", action="store_true")
    p.set_defaults(func=_cmd_ddl)

    p = sub.add_parser("compare-sql", help="SQL сверки прода с теневым прогоном")
    p.add_argument("dir")
    p.add_argument("--key", help="ключ через запятую; по умолчанию source.checks.unique_key")
    p.add_argument("--tol", type=float, default=1e-9)
    p.set_defaults(func=_cmd_compare_sql)
    return parser


def main(argv=None) -> int:
    # Консоль Windows (cp1251) не умеет часть символов (→ и т.п.): заменяем, а не падаем.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PipelineError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
