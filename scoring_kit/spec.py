"""Схема pipeline.yaml и её валидация."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Значения, которые точно принимает платформа (взяты из DAG'а старого инструмента).
DEFAULT_FLAVOR = "16cpu-256ram"
DEFAULT_TIME_LIMIT = "7d"

IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
TABLE_RE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$")
DAG_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
PG_IDENT_MAX = 63

STG_SUFFIX = "_stg"
SHADOW_SUFFIX = "_shadow"
# Служебные колонки, которые приёмник может добавить при загрузке из stg.
SCORED_AT_COLUMN = "scored_at"
MODEL_VERSION_COLUMN = "model_mrid"


class PipelineError(Exception):
    """Ошибка конфигурации пайплайна с понятным пользователю текстом."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_idents(names: list[str], what: str) -> list[str]:
    bad = [n for n in names if not IDENT_RE.match(n)]
    if bad:
        raise ValueError(f"{what}: недопустимые имена {bad} (только [a-z0-9_], с буквы или _)")
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"{what}: повторяются {dups}")
    return names


class WaitFor(_Strict):
    tables: list[str] = Field(min_length=1)
    timeout_seconds: int = 83000


class SourceChecks(_Strict):
    min_rows: int = Field(default=1, ge=0)
    unique_key: list[str] = []

    @field_validator("unique_key")
    @classmethod
    def _key(cls, v):
        return _check_idents(v, "source.checks.unique_key")


class Source(_Strict):
    query: str
    flavor: str = DEFAULT_FLAVOR
    checks: SourceChecks = SourceChecks()

    @field_validator("query")
    @classmethod
    def _query(cls, v: str):
        v = v.strip()
        if not v:
            raise ValueError("source.query пустой")
        return v


class Model(_Strict):
    mrid: list[str] = Field(min_length=1)
    image: str
    requirements: list[str] = []
    predictor: str = "predictor.py"
    predictor_class: str = "Predictor"
    flavor: str = DEFAULT_FLAVOR
    batch_size: int = Field(default=100_000, gt=0)
    # Порядок features = порядок колонок, на которых обучалась модель.
    features: list[str] = Field(min_length=1)
    cat_features: list[str] = []
    num_dtype: Literal["float32", "float64"] = "float64"
    # category — для LightGBM (pandas category), str — для CatBoost (строки, пропуск = "nan").
    cat_as: Literal["category", "str"] = "category"
    score_column: str = "score"
    # null — не проверять диапазон скора.
    score_range: Optional[tuple[float, float]] = (0.0, 1.0)

    @field_validator("mrid", mode="before")
    @classmethod
    def _mrid(cls, v):
        return [v] if isinstance(v, str) else v

    @field_validator("features")
    @classmethod
    def _features(cls, v):
        return _check_idents(v, "model.features")

    @field_validator("score_column")
    @classmethod
    def _score_column(cls, v):
        _check_idents([v], "model.score_column")
        return v

    @model_validator(mode="after")
    def _cats_subset(self):
        _check_idents(self.cat_features, "model.cat_features")
        extra = [c for c in self.cat_features if c not in self.features]
        if extra:
            raise ValueError(f"model.cat_features не входят в features: {extra}")
        if self.score_range is not None and self.score_range[0] > self.score_range[1]:
            raise ValueError("model.score_range: левая граница больше правой")
        return self


class Sink(_Strict):
    table: str
    # replace — атомарно: truncate + insert из stg в одной транзакции (гранты и view сохраняются).
    # append  — insert из stg.
    mode: Literal["replace", "append"] = "replace"
    columns: dict[str, str] = Field(min_length=1)
    add_scored_at: bool = False
    add_model_version: bool = False
    harmonize: bool = True
    actualize: bool = True
    flavor: str = DEFAULT_FLAVOR
    # Только для `scoring ddl`: ключ распределения при создании таблицы.
    distributed_by: list[str] = []

    @field_validator("table")
    @classmethod
    def _table(cls, v):
        if not TABLE_RE.match(v):
            raise ValueError(f"sink.table '{v}': нужен формат schema.table в нижнем регистре")
        name = v.split(".")[1]
        if len(name) + len(STG_SUFFIX) > PG_IDENT_MAX:
            raise ValueError(f"sink.table '{v}': имя с суффиксом {STG_SUFFIX} длиннее {PG_IDENT_MAX}")
        return v

    @field_validator("columns")
    @classmethod
    def _columns(cls, v: dict):
        _check_idents(list(v), "sink.columns")
        # «varchar » с пробелом встречался в старых пайплайнах.
        cleaned = {k: " ".join(str(t).split()) for k, t in v.items()}
        empty = [k for k, t in cleaned.items() if not t]
        if empty:
            raise ValueError(f"sink.columns: не указан тип для {empty}")
        return cleaned

    @model_validator(mode="after")
    def _service_columns(self):
        clash = [
            c
            for c, on in ((SCORED_AT_COLUMN, self.add_scored_at), (MODEL_VERSION_COLUMN, self.add_model_version))
            if on and c in self.columns
        ]
        if clash:
            raise ValueError(f"sink {self.table}: колонки {clash} добавляются автоматически, уберите их из columns")
        bad = [c for c in self.distributed_by if c not in self.columns]
        if bad:
            raise ValueError(f"sink {self.table}: distributed_by {bad} нет в columns")
        return self

    @property
    def short_name(self) -> str:
        return self.table.split(".")[1]

    @property
    def stg_table(self) -> str:
        return self.table + STG_SUFFIX


class Pipeline(_Strict):
    dag_id: str
    description: str = ""
    owner: str
    schedule: Optional[str] = None
    timezone: str = "UTC"
    start_date: dt.date = dt.date(2025, 1, 1)
    tags: list[str] = []
    gp_service: str
    gp_mode: str = "dal"
    time_limit: str = DEFAULT_TIME_LIMIT
    retries: int = Field(default=1, ge=0)
    retry_delay_minutes: int = Field(default=10, ge=0)
    wait_for: Optional[WaitFor] = None
    source: Source
    model: Model
    sinks: list[Sink] = Field(min_length=1)

    @field_validator("dag_id")
    @classmethod
    def _dag_id(cls, v):
        if not DAG_ID_RE.match(v):
            raise ValueError(f"dag_id '{v}': только буквы, цифры, _ . -")
        return v

    @model_validator(mode="after")
    def _cross(self):
        names = [s.short_name for s in self.sinks]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            raise ValueError(f"sinks: одинаковые имена таблиц {dups} — task_id совпадут")
        for s in self.sinks:
            if self.model.score_column not in s.columns:
                raise ValueError(
                    f"sink {s.table}: в columns нет колонки скора '{self.model.score_column}'"
                )
        return self

    @property
    def data_file(self) -> str:
        # BatchInferenceOperator читает вход только через pd.read_csv, поэтому между тасками всегда csv.
        return "data.csv"

    def as_shadow(self) -> "Pipeline":
        """Копия для теневого прогона: свой dag_id и таблицы с суффиксом _shadow.

        Актуализация выключена: на теневую таблицу никто не подписан, а засорять
        метаданные хранилища незачем. Гармонизация грантов остаётся, чтобы её читать.
        """
        sinks = []
        for s in self.sinks:
            table = s.table + SHADOW_SUFFIX
            if len(table.split(".")[1]) + len(STG_SUFFIX) > PG_IDENT_MAX:
                raise PipelineError(f"{table}{STG_SUFFIX}: имя длиннее {PG_IDENT_MAX} символов")
            sinks.append(s.model_copy(update={"table": table, "actualize": False}))
        return self.model_copy(
            update={"dag_id": self.dag_id + SHADOW_SUFFIX, "sinks": sinks, "tags": [*self.tags, "shadow"]}
        )


def _format_errors(err: ValidationError) -> str:
    lines = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "<корень>"
        msg = e["msg"].removeprefix("Value error, ")
        lines.append(f"  - {loc}: {msg}")
    return "\n".join(lines)


def load_pipeline(pipeline_dir: Union[str, Path]) -> Pipeline:
    """Читает и валидирует <pipeline_dir>/pipeline.yaml."""
    path = Path(pipeline_dir) / "pipeline.yaml"
    if not path.exists():
        raise PipelineError(f"Не найден {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise PipelineError(f"{path}: ошибка разбора YAML\n{e}") from e
    try:
        pipeline = Pipeline.model_validate(raw)
    except ValidationError as e:
        raise PipelineError(f"{path}: ошибки в конфиге\n{_format_errors(e)}") from e
    predictor = Path(pipeline_dir) / pipeline.model.predictor
    if not predictor.exists():
        raise PipelineError(f"{path}: не найден файл предиктора {predictor}")
    return pipeline
