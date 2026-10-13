"""Упаковка модели: модель + своя обработка + контракт фичей в одном объекте для Model Registry.

Упакованная модель — основной способ отдать модель в скоринг:

    from scoring_kit import pack_model

    def prepare(df):
        df["debt_log"] = np.log1p(df["debt_amt"])
        return df

    pack_model(model, features=[...], cat_features=[...], prepare=prepare,
               sample=X_raw.head(1000), path="model_packed.pkl")

Файл загружается в Model Registry, в pipeline.yaml у модели ставится `packed: true`. Обработка
едет вместе с моделью, поэтому на скоринге она гарантированно та же, что заложил автор, и
одинакова в обоих движках: в Greenplum предиктор вызывает `predict`, в DLH его вызывает
DlhBatchInferenceOperator (список входных колонок он берёт из атрибута `features`).

Как это устроено:
- LightGBM и CatBoost хранятся в переносимом виде (текст модели / байты), а не pickle объекта:
  версии библиотек в поде и в Spark-образе могут отличаться от обучения; остальные модели — pickle;
- обработка хранится исходным текстом функции и выполняется в чистом пространстве имён (np, pd),
  поэтому не зависит от ноутбука, где её написали;
- при распаковке объект восстанавливает себя из исходника, встроенного в pickle
  (`__reduce__` → `eval`): для загрузки нужны только pandas/numpy и библиотека модели.

Модуль не зависит от остального пакета: `scoring packer` выгружает его одним файлом для сред,
куда scoring-kit не установить.
"""

from __future__ import annotations

import ast
import base64
import builtins
import inspect
import pickle
import symtable
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

# В автономной копии (`scoring packer`) сюда подставлен исходник хелперов приведения типов.
_HELPERS_SOURCE = None

_CLASS_TEMPLATE = '''
class ScoringBundle:
    """Упакованная модель: обработка + контракт фичей + модель (сгенерировано scoring-kit)."""

    scoring_kit_packed = True

    def __init__(self, state):
        self.state = state
        # Входные колонки (до обработки): по этому атрибуту оператор DLH отбирает колонки источника.
        self.features = list(state.get("input_columns") or state["features"])
        self._model = None
        self._prepare = None

    def _loaded(self):
        if self._model is None:
            kind = self.state["model_kind"]
            if kind == "lightgbm":
                import lightgbm as lgb

                self._model = lgb.Booster(model_str=self.state["model"])
            elif kind == "catboost":
                import base64
                import tempfile

                import catboost

                with tempfile.NamedTemporaryFile(suffix=".cbm") as f:
                    f.write(base64.b64decode(self.state["model"]))
                    f.flush()
                    model = catboost.CatBoost()
                    model.load_model(f.name)
                self._model = model
            elif kind == "pickle":
                import base64
                import pickle

                self._model = pickle.loads(base64.b64decode(self.state["model"]))
            else:
                raise ValueError(f"неизвестный тип модели {kind}")
        return self._model

    def _prepare_func(self):
        source = self.state.get("prepare_source")
        if not source:
            return None
        if self._prepare is None:
            import numpy as np
            import pandas as pd

            namespace = {"np": np, "pd": pd}
            exec(compile(source, "<prepare>", "exec"), namespace)
            self._prepare = namespace[self.state["prepare_name"]]
        return self._prepare

    def predict(self, df):
        import numpy as np
        import pandas as pd

__HELPERS__
        s = self.state
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(
                f"[scoring-kit] упакованной модели нужны колонки {missing}, в выборке их нет. "
                f"Есть: {list(df.columns)}"
            )
        data = df[self.features].copy()
        prepare = self._prepare_func()
        if prepare is not None:
            data = sk_normalize_decimals(data)
            index = data.index
            data = prepare(data)
            if not isinstance(data, pd.DataFrame):
                raise ValueError("[scoring-kit] обработка (prepare) должна вернуть DataFrame")
            if len(data) != len(index) or not data.index.equals(index):
                raise ValueError(
                    "[scoring-kit] обработка (prepare) не должна удалять, добавлять или переставлять строки: "
                    f"пришло {len(index)}, вернулось {len(data)}"
                )
            lost = [c for c in s["features"] if c not in data.columns]
            if lost:
                raise ValueError(f"[scoring-kit] после обработки (prepare) нет фичей модели {lost}")
        if s.get("cast", True):
            X = sk_prepare_features(data, s["features"], s["cat_features"], s["num_dtype"], s["cat_as"])
        else:
            X = data
        X = X[s["features"]]
        model = self._loaded()
        proba = s["output_kind"] == "proba"
        if s["model_kind"] == "catboost":
            values = np.asarray(model.predict(X, prediction_type="Probability" if proba else "RawFormulaVal"))
        elif proba and hasattr(model, "predict_proba"):
            values = np.asarray(model.predict_proba(X))
        else:
            values = np.asarray(model.predict(X))
        values = values[:, 1] if values.ndim == 2 else values
        result = pd.DataFrame({s["score_column"]: values.astype("float64")}, index=df.index)
        # Рядом со скором — признаки после обработки (до приведения типов): процесс может записать
        # их в витрину, перечислив в model.output_columns. Скор всегда первой колонкой.
        for column in data.columns:
            if column != s["score_column"]:
                result[column] = data[column].to_numpy()
        return result

    def describe(self):
        """Что внутри упаковки: для ревью и для проверки перед загрузкой в реестр."""
        s = self.state
        pickled = " (pickle: версии библиотек должны быть как при обучении)" if s["model_kind"] == "pickle" else ""
        result = "вероятность" if s["output_kind"] == "proba" else "число (predict)"
        if s.get("cast", True):
            cast = "числа -> " + s["num_dtype"] + ", категории -> " + s["cat_as"]
        else:
            cast = "выключено"
        lines = [
            "Упакованная модель scoring-kit",
            "  модель:            " + s["model_kind"] + pickled,
            "  что выдаёт:        " + result + " -> колонка " + s["score_column"],
            "  входные колонки:   " + str(len(self.features)) + ": " + ", ".join(self.features),
            "  фичи модели:       " + str(len(s["features"])) + ": " + ", ".join(s["features"]),
            "  категориальные:    " + (", ".join(s["cat_features"]) or "-"),
            "  приведение типов:  " + cast,
        ]
        if s.get("produced_columns"):
            lines.append("  можно писать в витрину (после обработки): " + ", ".join(s["produced_columns"]))
        if s.get("prepare_source"):
            lines += ["  обработка:", ""] + ["    " + line for line in s["prepare_source"].rstrip().splitlines()]
        else:
            lines.append("  обработка:         нет (только приведение типов)")
        return chr(10).join(lines)

    def __repr__(self):
        return "<ScoringBundle " + self.state["model_kind"] + ": " + str(len(self.features)) + " входных колонок>"

    def __reduce__(self):
        return (eval, (_rebuild_expression(self.state),))


def _rebuild_expression(state):
    # После exec исходник кладётся в пространство имён как _BUNDLE_SOURCE: так объект можно
    # сериализовать повторно (Spark отправляет модель на экзекьюторы).
    return (
        "(lambda src, ns: (exec(src, ns), ns.__setitem__('_BUNDLE_SOURCE', src), ns['ScoringBundle']("
        + repr(state)
        + "))[2])("
        + repr(_BUNDLE_SOURCE)
        + ", {})"
    )
'''


class PackError(ValueError):
    """Модель нельзя упаковать: сообщение объясняет, что поправить."""


def _helpers_source() -> str:
    if _HELPERS_SOURCE is not None:
        return _HELPERS_SOURCE
    from scoring_kit import runtime

    return inspect.getsource(runtime.sk_normalize_decimals) + "\n" + inspect.getsource(runtime.sk_prepare_features)


def bundle_source() -> str:
    """Исходник упакованной модели: класс ScoringBundle и функция пересборки (без зависимостей от пакета)."""
    helpers = textwrap.indent(_helpers_source(), " " * 8)
    return _CLASS_TEMPLATE.replace("__HELPERS__", helpers.rstrip("\n"))


def standalone_source() -> str:
    """Этот модуль одним файлом: для сред, куда scoring-kit не установить (`scoring packer`)."""
    text = Path(__file__).read_text(encoding="utf-8")
    marker = "_HELPERS_SOURCE = None\n"
    if text.count(marker) != 1:
        raise RuntimeError("bundle.py: не найдена строка _HELPERS_SOURCE")
    return text.replace(marker, f"_HELPERS_SOURCE = {_helpers_source()!r}\n")


def _materialize(state: dict) -> Any:
    src = bundle_source()
    namespace: dict = {}
    exec(compile(src, "scoring_bundle", "exec"), namespace)
    namespace["_BUNDLE_SOURCE"] = src
    return namespace["ScoringBundle"](state)


def _model_state(model: Any) -> tuple:
    """(kind, представление модели): бустинги — переносимо, остальное — pickle."""
    booster = getattr(model, "booster_", None)
    if booster is None and type(model).__name__ == "Booster" and hasattr(model, "model_to_string"):
        booster = model
    if booster is not None:
        best = getattr(model, "best_iteration_", None) or getattr(booster, "best_iteration", 0) or 0
        num_iteration = best if best and best > 0 else -1
        return "lightgbm", booster.model_to_string(num_iteration=num_iteration)
    if hasattr(model, "save_model") and type(model).__module__.startswith("catboost"):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.cbm"
            model.save_model(str(path))
            return "catboost", base64.b64encode(path.read_bytes()).decode("ascii")
    if getattr(model, "scoring_kit_packed", False):
        raise PackError("эта модель уже упакована: передайте исходную модель, а не результат pack_model")
    if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
        raise PackError(
            f"у объекта {type(model).__module__}.{type(model).__name__} нет predict / predict_proba — это не модель"
        )
    try:
        payload = pickle.dumps(model)
    except Exception as e:
        raise PackError(f"модель {type(model).__name__} не сериализуется: {e}") from e
    return "pickle", base64.b64encode(payload).decode("ascii")


# ---------------------------------------------------------------- обработка (prepare)

_PROVIDED = ("np", "pd")


def _prepare_source(prepare: Union[Callable, str, Path, None]) -> tuple:
    """(имя функции, исходник). Исходник должен работать сам по себе: только np, pd и свои импорты."""
    if prepare is None:
        return None, None
    name = None
    if callable(prepare):
        name = getattr(prepare, "__name__", "")
        if name == "<lambda>":
            raise PackError("prepare: нужна обычная функция через def, а не lambda")
        try:
            source = inspect.getsource(prepare)
        except (OSError, TypeError) as e:
            raise PackError(
                "prepare: не удалось получить текст функции. Передайте её текстом (строкой с def ...) "
                "или путём к .py файлу"
            ) from e
    else:
        text = str(prepare)
        looks_like_path = "\n" not in text and text.strip().endswith(".py")
        source = Path(text).read_text(encoding="utf-8") if looks_like_path else text
    source = textwrap.dedent(source).strip() + "\n"
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise PackError(f"prepare: ошибка в коде, строка {e.lineno}: {e.msg}") from e

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    allowed = (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.Expr)
    extra = [n for n in tree.body if not isinstance(n, allowed)]
    if extra:
        raise PackError(
            f"prepare: строка {extra[0].lineno} — на верхнем уровне допустимы только import, константы и функции"
        )
    if name is None:
        named = [f for f in funcs if f.name == "prepare"]
        if not named and len(funcs) != 1:
            raise PackError("prepare: в коде должна быть функция def prepare(df)")
        name = (named or funcs)[0].name
    entry = next((f for f in funcs if f.name == name), None)
    if entry is None:
        raise PackError(f"prepare: в коде нет функции {name}")
    if entry.decorator_list:
        raise PackError("prepare: уберите декораторы с функции")
    if len(entry.args.args) < 1:
        raise PackError(f"prepare: функция {name} должна принимать датафрейм: def {name}(df)")

    unknown = sorted(_free_names(source) - set(_PROVIDED) - set(dir(builtins)))
    if unknown:
        raise PackError(
            f"prepare: функция использует {unknown}, а они определены вне её (в ноутбуке). "
            "В упаковке их не будет: перенесите определения внутрь функции или добавьте import внутри неё. "
            "Доступны без импорта: np, pd"
        )
    return name, source


def _free_names(source: str) -> set:
    """Имена, которые код берёт из глобального пространства, но сам на верхнем уровне не определяет."""
    top = symtable.symtable(source, "<prepare>", "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    free: set = set()

    def walk(table):
        for sym in table.get_symbols():
            if table is top:
                if sym.is_referenced() and not (sym.is_assigned() or sym.is_imported() or sym.is_namespace()):
                    free.add(sym.get_name())
            elif sym.is_global() and sym.is_referenced():
                free.add(sym.get_name())
        for child in table.get_children():
            walk(child)

    walk(top)
    return free - defined


# ---------------------------------------------------------------- упаковка


def _state(model: Any, *, features, cat_features, num_dtype, cat_as, cast, output_kind, score_column,
           prepare, input_columns, source) -> dict:
    features = [str(f) for f in features]
    cat_features = [str(f) for f in cat_features]
    if not features:
        raise PackError("features: перечислите фичи модели в том порядке, в котором она обучалась")
    if len(set(features)) != len(features):
        dups = sorted({f for f in features if features.count(f) > 1})
        raise PackError(f"features: повторы {dups}")
    extra = [c for c in cat_features if c not in features]
    if extra:
        raise PackError(f"cat_features не входят в features: {extra}")
    if output_kind not in ("proba", "predict"):
        raise PackError("output_kind: 'proba' (вероятность) или 'predict' (число)")
    if num_dtype not in ("float32", "float64"):
        raise PackError("num_dtype: 'float32' или 'float64'")
    kind, payload = _model_state(model)
    if cat_as is None:
        cat_as = "str" if kind == "catboost" else "category"
    if cat_as not in ("category", "str"):
        raise PackError("cat_as: 'category' (LightGBM) или 'str' (CatBoost)")
    if kind == "lightgbm" and cast and cat_features and cat_as != "category":
        raise PackError("для LightGBM категориальные фичи — cat_as='category'")
    prepare_name, prepare_source = _prepare_source(prepare)
    return {
        "format": 2,
        "model_kind": kind,
        "model": payload,
        "input_columns": [str(c) for c in input_columns] if input_columns is not None else None,
        "prepare_name": prepare_name,
        "prepare_source": prepare_source,
        "features": features,
        "cat_features": cat_features,
        "num_dtype": num_dtype,
        "cat_as": cat_as,
        "cast": bool(cast),
        "score_column": score_column,
        "output_kind": output_kind,
        "source_mrid": list(source or []),
    }


def _infer_input_columns(state: dict, sample) -> list:
    """Какие колонки выборки на самом деле нужны: убираем по одной и смотрим, сломается ли расчёт."""
    columns = [str(c) for c in sample.columns]
    head = sample.head(50)
    try:
        _materialize(dict(state, input_columns=columns)).predict(head)
    except Exception as e:
        raise PackError(f"обработка (prepare) не отработала на sample: {type(e).__name__}: {e}") from e
    needed = []
    for column in columns:
        probe = dict(state, input_columns=[c for c in columns if c != column])
        try:
            _materialize(probe).predict(head)
        except Exception:
            needed.append(column)
    return needed


def pack_model(
    model: Any,
    features: Iterable[str],
    cat_features: Iterable[str] = (),
    prepare: Union[Callable, str, Path, None] = None,
    *,
    sample=None,
    input_columns: Optional[Iterable[str]] = None,
    output_kind: str = "proba",
    num_dtype: str = "float64",
    cat_as: Optional[str] = None,
    cast: bool = True,
    score_column: str = "score",
    path: Union[str, Path, None] = "model_packed.pkl",
    verbose: bool = True,
):
    """Упаковывает модель вместе с обработкой и контрактом фичей для Model Registry.

    model         обученная модель (LightGBM, CatBoost или любая с predict / predict_proba).
    features      фичи модели в том порядке, в котором она обучалась (после обработки).
    cat_features  категориальные из features.
    prepare       своя обработка: функция `def prepare(df): ...; return df` (или её текст, или путь
                  к .py). Получает сырые колонки выборки, возвращает датафрейм с фичами модели.
                  Строки не удаляет и не переставляет. Доступны np и pd; остальное импортируется
                  внутри функции.
    sample        несколько сотен строк сырой выборки (как из SQL процесса). С ним упаковка
                  проверяет себя: считает скоры по шагам и через упакованный объект и сравнивает.
    input_columns сырые колонки, нужные обработке. Без обработки это features; с обработкой
                  определяются по sample или задаются явно.
    output_kind   'proba' — вероятность класса 1, 'predict' — число.
    num_dtype     'float32', если при обучении числа приводили к float32.
    cat_as        'category' (LightGBM) или 'str' (CatBoost); по умолчанию по типу модели.
    cast          False — не приводить типы автоматически (всё делает prepare).
    path          куда сохранить файл; None — не сохранять.

    Возвращает упакованный объект: `.predict(df)` -> DataFrame со скором и признаками после обработки,
    `.describe()` -> что внутри.
    """
    state = _state(
        model, features=list(features), cat_features=list(cat_features), num_dtype=num_dtype, cat_as=cat_as,
        cast=cast, output_kind=output_kind, score_column=score_column, prepare=prepare,
        input_columns=input_columns, source=None,
    )
    if state["prepare_source"] and state["input_columns"] is None:
        if sample is None:
            raise PackError(
                "с обработкой (prepare) нужно знать сырые колонки выборки: передайте sample=<несколько сотен строк "
                "сырой выборки> или input_columns=[...]"
            )
        state["input_columns"] = _infer_input_columns(state, sample)
        if not state["input_columns"]:
            raise PackError("не удалось определить входные колонки по sample: задайте input_columns=[...] явно")
    if sample is not None:
        try:
            # какие колонки модель отдаёт рядом со скором: их имена можно указать в процессе для записи в витрину
            state["produced_columns"] = [str(c) for c in _materialize(state).predict(sample.head(20)).columns[1:]]
        except Exception:
            pass          # понятную ошибку даст самопроверка ниже
    # Дальше работаем только с тем, что реально восстановится из файла.
    payload = pickle.dumps(_materialize(state))
    restored = pickle.loads(payload)

    checked = _self_check(model, state, restored, sample, prepare) if sample is not None else None
    if path is not None:
        Path(path).write_bytes(payload)
    if verbose:
        print(restored.describe())
        print()
        if checked is None:
            print("Проверка не выполнялась: передайте sample=..., чтобы сравнить скоры до и после упаковки.")
        else:
            for line in checked:
                print(line)
        if path is not None:
            print(f"Файл: {path} ({max(1, len(payload) // 1024)} КБ). Загрузите его в Model Registry; "
                  "в процессе отметьте, что модель упакована (packed: true).")
    return restored


def _self_check(model, state: dict, restored, sample, prepare) -> list:
    """Скоры упакованного объекта против расчёта по шагам на исходных объектах из ноутбука."""
    import numpy as np

    missing = [c for c in restored.features if c not in sample.columns]
    if missing:
        raise PackError(f"в sample нет входных колонок {missing}")
    column = state["score_column"]
    try:
        packed = restored.predict(sample)[column].to_numpy()
    except Exception as e:
        raise PackError(f"упакованная модель не отработала на sample: {type(e).__name__}: {e}") from e
    report = [f"Проверка на sample: {len(sample)} строк, упакованная модель отработала."]

    def compare(other, what):
        same_nan = bool((np.isnan(packed) == np.isnan(other)).all())
        diff = float(np.nanmax(np.abs(packed - other))) if len(other) and not np.isnan(packed).all() else 0.0
        if not same_nan or diff > 1e-9:
            raise PackError(
                f"самопроверка не прошла: скоры упакованной модели расходятся с {what} (макс. разница {diff:.3g})"
            )
        report.append(f"  совпадает с {what}: макс. разница {diff:.1e}")

    if state["prepare_source"] and not callable(prepare):
        return report      # обработка передана текстом: независимого расчёта для сравнения нет
    data = prepare(sample.copy()) if callable(prepare) else sample
    # 1. Та же модель и те же типы, но обработка — живая функция из ноутбука на всей выборке.
    reference = _materialize(dict(state, prepare_source=None, prepare_name=None, input_columns=None))
    compare(reference.predict(data)[column].to_numpy(), "расчётом по шагам (функция из ноутбука)")
    # 2. Исходный объект модели на тех же данных — если у него обычный интерфейс.
    try:
        namespace: dict = {}
        exec(_helpers_source(), namespace)
        if state["cast"]:
            X = namespace["sk_prepare_features"](
                data, state["features"], state["cat_features"], state["num_dtype"], state["cat_as"])
        else:
            X = data
        X = X[state["features"]]
        if state["output_kind"] == "proba" and hasattr(model, "predict_proba"):
            original = np.asarray(model.predict_proba(X))
        elif state["output_kind"] == "predict" or type(model).__name__ == "Booster":
            original = np.asarray(model.predict(X))
        else:
            original = None
        if original is not None:
            original = (original[:, 1] if original.ndim == 2 else original).astype("float64")
    except Exception:
        original = None
    if original is not None:
        compare(original, "исходной моделью")
    return report


# ---------------------------------------------------------------- упаковка по pipeline.yaml (CLI)


def build_bundle(model_obj: Any, contract: Any, prepare: Union[Callable, str, Path, None] = None,
                 input_columns: Optional[Iterable[str]] = None) -> Any:
    """Упаковка по контракту из pipeline.yaml (`scoring bundle`)."""
    if contract.output_kind is None:
        raise PackError("scoring bundle: в model нужен output_kind (proba | predict)")
    state = _state(
        model_obj, features=contract.features, cat_features=contract.cat_features, num_dtype=contract.num_dtype,
        cat_as=contract.cat_as, cast=True, output_kind=contract.output_kind, score_column=contract.score_column,
        prepare=prepare, input_columns=input_columns, source=contract.mrid,
    )
    if state["prepare_source"] and state["input_columns"] is None:
        raise PackError("scoring bundle: с обработкой укажите входные колонки (--input-columns a,b,c)")
    return _materialize(state)


def bundle_from_file(model_path: Union[str, Path], contract: Any, prepare=None, input_columns=None) -> Any:
    from scoring_kit import runtime

    return build_bundle(runtime.sk_load_model(str(model_path)), contract, prepare, input_columns)


def load_packed(path: Union[str, Path]) -> Any:
    """Читает упакованную модель из файла."""
    try:
        obj = pickle.loads(Path(path).read_bytes())
    except Exception:
        import joblib

        obj = joblib.load(path)
    if not getattr(obj, "scoring_kit_packed", False):
        raise PackError(f"{path}: это не упакованная модель scoring-kit")
    return obj
