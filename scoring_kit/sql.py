"""SQL, который генерирует scoring-kit."""

from __future__ import annotations

from scoring_kit.spec import MODEL_VERSION_COLUMN, SCORED_AT_COLUMN, Pipeline, Sink


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def service_columns(sink: Sink, pipeline: Pipeline) -> list[tuple[str, str, str]]:
    """(имя, тип, выражение) служебных колонок, которые добавляются при загрузке."""
    cols = []
    if sink.add_scored_at:
        cols.append((SCORED_AT_COLUMN, "timestamp", "now()"))
    if sink.add_model_version:
        cols.append((MODEL_VERSION_COLUMN, "varchar", _literal(",".join(pipeline.model.mrid))))
    return cols


def load_sql(sink: Sink, pipeline: Pipeline) -> str:
    """Перенос stg -> целевая таблица одной транзакцией.

    replace: truncate + insert. В отличие от drop/create таблица не исчезает,
    гранты и зависимые view сохраняются, а читатели до commit видят старые данные
    (truncate берёт эксклюзивную блокировку — запросы подождут секунды загрузки).
    """
    extra = service_columns(sink, pipeline)
    target_cols = list(sink.columns) + [name for name, _, _ in extra]
    select_exprs = list(sink.columns) + [expr for _, _, expr in extra]
    # Целевая таблица создаётся при первом запуске по sinks[].columns; существующую не трогаем.
    parts = ["begin;", ddl(sink, pipeline, if_not_exists=True)]
    if sink.mode == "replace":
        parts.append(f"truncate table {sink.table};")
    parts.append(
        f"insert into {sink.table} ({', '.join(target_cols)})\n"
        f"select {', '.join(select_exprs)}\n"
        f"from {sink.stg_table};"
    )
    parts.append("commit;")
    return "\n".join(parts)


def harmonize_sql(sink: Sink) -> str:
    return f"select public.tcs_harmonize_grants('{sink.table}')"


def actualize_sql(sink: Sink) -> str:
    return f"select public.ulabs_actualize('{sink.table}')"


def ddl(sink: Sink, pipeline: Pipeline, if_not_exists: bool = False) -> str:
    """create table для целевой таблицы (DAG выполняет его сам с if not exists)."""
    cols = list(sink.columns.items()) + [(n, t) for n, t, _ in service_columns(sink, pipeline)]
    width = max(len(n) for n, _ in cols)
    body = ",\n".join(f"    {n.ljust(width)} {t}" for n, t in cols)
    dist = (
        f"distributed by ({', '.join(sink.distributed_by)})"
        if sink.distributed_by
        else "distributed randomly"
    )
    return f"create table {sink.table} (\n{body}\n)\n{dist};"


def compare_sql(prod_table: str, shadow_table: str, key: list[str], score_column: str, tol: float) -> str:
    """Сверка прода с теневым прогоном: одна строка с итогами."""
    using = ", ".join(key)
    p0 = key[0]
    return (
        "select\n"
        f"    sum(case when s.{p0} is null then 1 else 0 end) as only_in_prod,\n"
        f"    sum(case when p.{p0} is null then 1 else 0 end) as only_in_shadow,\n"
        f"    sum(case when p.{p0} is not null and s.{p0} is not null then 1 else 0 end) as matched,\n"
        f"    max(abs(p.{score_column} - s.{score_column})) as max_abs_diff,\n"
        f"    sum(case when abs(p.{score_column} - s.{score_column}) > {tol} then 1 else 0 end) as n_diff_over_tol\n"
        f"from {prod_table} p\n"
        f"full outer join {shadow_table} s using ({using});"
    )
