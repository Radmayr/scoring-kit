"""Схема pipeline.yaml и её валидация.

Один формат на все рецепты: общая шапка + поля рецепта. Конфиги 0.x (без `recipe`)
читаются как `recipe: single_model`, `engine: gp`.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Значения, которые точно принимает платформа (взяты из DAG'ов старого инструмента).
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
# Колонка, которую DlhBatchInferenceOperator всегда добавляет в результат.
DLH_PROCESSED_COLUMN = "processed_dttm"
# Колонки отчёта калибровки (fit_apply, sinks[].data: report).
REPORT_COLUMNS = (
    "segment", "cohort", "status", "train_cohorts", "train_rows", "test_rows", "apply_rows",
    "brier_raw", "brier_calibrated",
)

Recipe = Literal["single_model", "multi_model", "fit_apply"]
Engine = Literal["gp", "dlh"]


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


def _check_table(v: str, what: str) -> str:
    if not TABLE_RE.match(v):
        raise ValueError(f"{what} '{v}': нужен формат schema.table в нижнем регистре")
    return v


# ---------------------------------------------------------------- общие блоки


class WaitFor(_Strict):
    tables: list[str] = Field(min_length=1)
    timeout_seconds: int = 83000


class Alerts(_Strict):
    # Каналы ("#имя") и логины в корпоративном мессенджере.
    recipients: list[str] = Field(min_length=1)
    on_retry: bool = False
    message: str = ""


class SourceChecks(_Strict):
    min_rows: int = Field(default=1, ge=0)
    unique_key: list[str] = []

    @field_validator("unique_key")
    @classmethod
    def _key(cls, v):
        return _check_idents(v, "checks.unique_key")


class Source(_Strict):
    # Ровно одно из двух: произвольный SQL или готовая таблица.
    query: Optional[str] = None
    table: Optional[str] = None
    flavor: str = DEFAULT_FLAVOR
    checks: SourceChecks = SourceChecks()

    @field_validator("query")
    @classmethod
    def _query(cls, v):
        if v is not None and not v.strip():
            raise ValueError("source.query пустой")
        return v.strip() if v else v

    @field_validator("table")
    @classmethod
    def _table(cls, v):
        return _check_table(v, "source.table") if v else v

    @model_validator(mode="after")
    def _one(self):
        if (self.query is None) == (self.table is None):
            raise ValueError("source: укажите ровно одно из query или table")
        return self

    @property
    def sql(self) -> str:
        return self.query if self.query else f"select * from {self.table}"


class Model(_Strict):
    """Модель для single_model и элемент списка models в multi_model."""

    name: Optional[str] = None
    mrid: list[str] = Field(min_length=1)
    image: Optional[str] = None
    requirements: list[str] = []
    # Свой предиктор (файл) или встроенный (output_kind). Для multi_model — только встроенный.
    predictor: str = "predictor.py"
    predictor_class: str = "Predictor"
    output_kind: Optional[Literal["proba", "predict"]] = None
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
    # Колонки приёмников (кроме score_column), которые добавляет сам predict: их не ищем во входной выборке.
    output_columns: list[str] = []

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
        _check_idents(self.output_columns, "model.output_columns")
        extra = [c for c in self.cat_features if c not in self.features]
        if extra:
            raise ValueError(f"model.cat_features не входят в features: {extra}")
        if self.score_range is not None and self.score_range[0] > self.score_range[1]:
            raise ValueError("model.score_range: левая граница больше правой")
        if self.name is not None:
            _check_idents([self.name], "model.name")
        return self

    @property
    def builtin(self) -> bool:
        return self.output_kind is not None

    @property
    def runtime_key(self) -> tuple:
        """Модели с одинаковым окружением и ресурсами считаются одним job'ом."""
        return (self.image, tuple(self.requirements), self.flavor, self.batch_size)


class Sink(_Strict):
    table: str
    # Что писать: результат скоринга или (для fit_apply) отчёт калибровки.
    data: Literal["result", "report"] = "result"
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
        _check_table(v, "sink.table")
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


# ---------------------------------------------------------------- DLH


class DlhSettings(_Strict):
    image: str
    max_executors: Optional[int] = Field(default=None, gt=0, le=100)
    max_wait_seconds: Optional[int] = 3 * 3600
    # Куда положить выборку, если source задан запросом (оператор читает только таблицу).
    staging_table: Optional[str] = None

    @field_validator("staging_table")
    @classmethod
    def _staging(cls, v):
        return _check_table(v, "dlh.staging_table") if v else v


# ---------------------------------------------------------------- fit_apply


class FitApplyInput(_Strict):
    query: str
    flavor: str = "4cpu-16ram"
    min_rows: int = Field(default=1, ge=0)
    key: list[str] = []

    @field_validator("key")
    @classmethod
    def _key(cls, v):
        return _check_idents(v, "key")


class Job(_Strict):
    """Окружение пода для обучения на лету (BatchInferenceOperator с mrid=[])."""

    image: str
    requirements: list[str] = []
    flavor: str = "4cpu-16ram"


class Calibrate(_Strict):
    method: Literal["isotonic", "sigmoid"] = "isotonic"
    score: str
    target: str
    output: str
    # {поле: [значение | [значения группы], ...]}; несколько полей — декартово произведение.
    segments: dict[str, list[Union[str, int, list[Union[str, int]]]]] = Field(min_length=1)
    cohort_column: str
    # Для когорты на позиции i (в отсортированном списке когорт сегмента в dev) обучение идёт
    # на когортах с позициями i+offset. [-3, -2] = как в текущем процессе.
    train_offsets: list[int] = [-3, -2]
    # Что писать, если строка apply не откалибровалась (нет когорты/сегмента).
    uncalibrated: Literal["null", "copy_score"] = "null"

    @field_validator("uncalibrated", mode="before")
    @classmethod
    def _null(cls, v):
        # `uncalibrated: null` без кавычек YAML читает как None
        return "null" if v is None else v

    @model_validator(mode="after")
    def _check(self):
        _check_idents([self.score, self.target, self.output, self.cohort_column, *self.segments], "calibrate")
        if not self.train_offsets or any(o >= 0 for o in self.train_offsets):
            raise ValueError("calibrate.train_offsets: нужны отрицательные сдвиги, например [-3, -2]")
        return self

    @property
    def first_position(self) -> int:
        return -min(self.train_offsets)


# ---------------------------------------------------------------- пайплайн


class Pipeline(_Strict):
    recipe: Recipe = "single_model"
    engine: Engine = "gp"
    dag_id: str
    description: str = ""
    owner: str
    domain: Optional[str] = None
    schedule: Optional[str] = None
    timezone: str = "UTC"
    start_date: dt.date = dt.date(2025, 1, 1)
    tags: list[str] = []
    gp_service: str = "vrcl"
    gp_mode: str = "dal"
    time_limit: str = DEFAULT_TIME_LIMIT
    retries: int = Field(default=1, ge=0)
    retry_delay_minutes: int = Field(default=10, ge=0)
    alerts: Optional[Alerts] = None
    wait_for: Optional[WaitFor] = None

    # single_model / multi_model
    source: Optional[Source] = None
    model: Optional[Model] = None
    models: Optional[list[Model]] = None
    dlh: Optional[DlhSettings] = None

    # fit_apply
    dev: Optional[FitApplyInput] = None
    apply: Optional[FitApplyInput] = None
    job: Optional[Job] = None
    calibrate: Optional[Calibrate] = None

    sinks: list[Sink] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _model_defaults(cls, raw):
        # model_defaults — общие поля для всех моделей multi_model; модель может переопределить.
        if isinstance(raw, dict) and "model_defaults" in raw:
            raw = dict(raw)
            defaults = raw.pop("model_defaults") or {}
            raw["models"] = [{**defaults, **m} for m in raw.get("models") or []]
        return raw

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
        getattr(self, f"_check_{self.recipe}")()
        if self.engine == "dlh" and self.recipe != "single_model":
            raise ValueError("engine: dlh пока поддерживается только для recipe: single_model")
        return self

    def _need(self, *fields: str):
        missing = [f for f in fields if getattr(self, f) is None]
        if missing:
            raise ValueError(f"recipe {self.recipe}: нужны поля {missing}")

    def _forbid(self, *fields: str):
        extra = [f for f in fields if getattr(self, f) is not None]
        if extra:
            raise ValueError(f"recipe {self.recipe}: поля {extra} здесь не используются")

    def _result_sinks(self) -> list[Sink]:
        return [s for s in self.sinks if s.data == "result"]

    def _check_single_model(self):
        self._need("source", "model")
        self._forbid("models", "dev", "apply", "job", "calibrate")
        if any(s.data != "result" for s in self.sinks):
            raise ValueError("sinks[].data: report есть только у recipe: fit_apply")
        for s in self.sinks:
            if self.model.score_column not in s.columns:
                raise ValueError(f"sink {s.table}: в columns нет колонки скора '{self.model.score_column}'")
        if self.engine == "dlh":
            self._need("dlh")
            m = self.model
            if len(m.mrid) != 1:
                raise ValueError("engine: dlh — ровно одна модель (mrid упакованной модели, см. scoring bundle)")
            if not m.builtin:
                raise ValueError("engine: dlh — свой predictor.py не поддерживается; укажите model.output_kind")
            if m.output_columns:
                raise ValueError("engine: dlh — model.output_columns не поддерживаются")
            if len(self.sinks) != 1 or self.sinks[0].mode != "replace":
                raise ValueError("engine: dlh — один приёмник в режиме replace (оператор пишет createOrReplace)")
            s = self.sinks[0]
            if s.add_scored_at or s.add_model_version:
                raise ValueError("engine: dlh — add_scored_at/add_model_version не поддерживаются (есть processed_dttm)")
            if DLH_PROCESSED_COLUMN in s.columns:
                raise ValueError(f"engine: dlh — {DLH_PROCESSED_COLUMN} добавляется оператором, уберите из columns")
            if self.source.query and not self.dlh.staging_table:
                raise ValueError("engine: dlh с source.query — укажите dlh.staging_table (оператор читает таблицу)")
        else:
            self._forbid("dlh")
            if self.source.table is not None:
                raise ValueError("engine: gp — укажите source.query (source.table — для engine: dlh)")
            if self.model.image is None:
                raise ValueError("model.image обязателен")

    def _check_multi_model(self):
        self._need("source", "models")
        self._forbid("model", "dev", "apply", "job", "calibrate", "dlh")
        if any(s.data != "result" for s in self.sinks):
            raise ValueError("sinks[].data: report есть только у recipe: fit_apply")
        if self.source.table is not None:
            raise ValueError("multi_model: укажите source.query")
        if len(self.models) < 2:
            raise ValueError("multi_model: нужно хотя бы две модели (для одной — single_model)")
        for i, m in enumerate(self.models):
            if m.name is None:
                raise ValueError(f"models[{i}]: нужно имя (name)")
            if len(m.mrid) != 1:
                raise ValueError(f"models[{i}] {m.name}: ровно один mrid на модель")
            if not m.builtin:
                raise ValueError(f"models[{i}] {m.name}: укажите output_kind (proba | predict)")
            if m.image is None:
                raise ValueError(f"models[{i}] {m.name}: нужен image (или model_defaults.image)")
        _check_idents([m.name for m in self.models], "models.name")
        _check_idents([m.score_column for m in self.models], "models.score_column")
        # Одна колонка не может быть категориальной в одной модели и числовой в другой.
        kinds: dict[str, tuple[str, str]] = {}
        for m in self.models:
            for f in m.features:
                kind = ("cat", m.cat_as) if f in m.cat_features else ("num", m.num_dtype)
                if kinds.setdefault(f, kind) != kind:
                    raise ValueError(f"фича {f}: разные типы в разных моделях ({kinds[f]} и {kind})")
        scores = {m.score_column for m in self.models}
        for s in self.sinks:
            missing = sorted(scores - set(s.columns))
            if missing:
                raise ValueError(f"sink {s.table}: нет колонок скоров {missing}")

    def _check_fit_apply(self):
        self._need("dev", "apply", "job", "calibrate")
        self._forbid("source", "model", "models", "dlh")
        if not self.apply.key:
            raise ValueError("apply.key: укажите ключ apply-выборки")
        if not self._result_sinks():
            raise ValueError("fit_apply: нужен приёмник результата (sinks[].data: result)")
        for s in self.sinks:
            if s.data == "report":
                extra = [c for c in s.columns if c not in REPORT_COLUMNS]
                if extra:
                    raise ValueError(f"sink {s.table}: в отчёте калибровки нет колонок {extra}; есть {list(REPORT_COLUMNS)}")
            elif self.calibrate.output not in s.columns:
                raise ValueError(f"sink {s.table}: нет колонки калиброванного скора '{self.calibrate.output}'")

    # ------------------------------------------------------------ удобства

    @property
    def data_file(self) -> str:
        # BatchInferenceOperator читает вход только через pd.read_csv, поэтому между тасками всегда csv.
        return "data.csv"

    @property
    def all_models(self) -> list[Model]:
        return self.models if self.models else ([self.model] if self.model else [])

    @property
    def output_columns(self) -> list[str]:
        """Колонки, которые создаёт инференс (их не ищем во входной выборке)."""
        if self.recipe == "fit_apply":
            return [self.calibrate.output]
        cols = []
        for m in self.all_models:
            cols += [m.score_column, *m.output_columns]
        return cols

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
        update = {"dag_id": self.dag_id + SHADOW_SUFFIX, "sinks": sinks, "tags": [*self.tags, "shadow"]}
        if self.dlh is not None and self.dlh.staging_table:
            update["dlh"] = self.dlh.model_copy(update={"staging_table": self.dlh.staging_table + SHADOW_SUFFIX})
        return self.model_copy(update=update)


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
    m = pipeline.model
    if m is not None and not m.builtin:
        predictor = Path(pipeline_dir) / m.predictor
        if not predictor.exists():
            raise PipelineError(f"{path}: не найден файл предиктора {predictor} (или задайте model.output_kind)")
    return pipeline
