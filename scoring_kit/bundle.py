"""Упаковка модели для engine: dlh (DlhBatchInferenceOperator).

Оператор DLH вызывает у загруженной модели `predict(df)` и берёт список фичей из атрибута
`features`; в Spark-образе другие версии библиотек (lightgbm 4.x, numpy 2, pandas 3) и нет
нашего пакета. Поэтому бандл:

- хранит модель в переносимом виде (текст модели LightGBM / байты CatBoost), а не pickle объекта;
- сам приводит типы фичей по контракту (тот же код, что в подах engine: gp);
- при распаковке восстанавливает себя из исходника, встроенного в pickle (`__reduce__` → `eval`),
  поэтому для загрузки нужны только pandas/numpy и библиотека модели — ни scoring-kit, ни cloudpickle.

Результат `scoring bundle` загружается в Model Registry новой версией; её mrid указывается в yaml.
"""

from __future__ import annotations

import base64
import inspect
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Union

from scoring_kit import runtime
from scoring_kit.spec import Model

_CLASS_TEMPLATE = '''
class ScoringBundle:
    """Модель + контракт фичей для DlhBatchInferenceOperator (сгенерировано scoring-kit)."""

    def __init__(self, state):
        self.state = state
        self.features = list(state["features"])
        self._model = None

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
            else:
                raise ValueError(f"неизвестный тип модели {kind}")
        return self._model

    def predict(self, df):
        import numpy as np
        import pandas as pd

__PREPARE__
        s = self.state
        X = sk_prepare_features(df, s["features"], s["cat_features"], s["num_dtype"], s["cat_as"])[s["features"]]
        model = self._loaded()
        if s["model_kind"] == "catboost":
            kind = "Probability" if s["output_kind"] == "proba" else "RawFormulaVal"
            values = np.asarray(model.predict(X, prediction_type=kind))
            values = values[:, 1] if values.ndim == 2 else values
        else:
            values = np.asarray(model.predict(X))
        return pd.DataFrame({s["score_column"]: values.astype("float64")}, index=df.index)

    def __reduce__(self):
        return (eval, (_rebuild_expression(self.state),))


def _rebuild_expression(state):
    # После exec исходник кладётся в пространство имён как _BUNDLE_SOURCE: так бандл можно
    # сериализовать повторно (Spark отправляет модель на экзекьюторы).
    return (
        "(lambda src, ns: (exec(src, ns), ns.__setitem__('_BUNDLE_SOURCE', src), ns['ScoringBundle']("
        + repr(state)
        + "))[2])("
        + repr(_BUNDLE_SOURCE)
        + ", {})"
    )
'''


def bundle_source() -> str:
    """Исходник бандла: класс ScoringBundle и функция пересборки (без зависимостей от пакета)."""
    prepare = textwrap.indent(inspect.getsource(runtime.sk_prepare_features), " " * 8)
    return _CLASS_TEMPLATE.replace("__PREPARE__", prepare.rstrip("\n"))


def _materialize(state: dict) -> Any:
    src = bundle_source()
    namespace: dict = {}
    exec(compile(src, "scoring_bundle", "exec"), namespace)
    namespace["_BUNDLE_SOURCE"] = src
    return namespace["ScoringBundle"](state)


def _model_state(model: Any) -> tuple[str, str]:
    """(kind, переносимое представление модели)."""
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
    raise ValueError(
        f"scoring bundle: неподдерживаемый тип модели {type(model).__module__}.{type(model).__name__} "
        "(поддерживаются LightGBM и CatBoost)"
    )


def build_bundle(model_obj: Any, contract: Model) -> Any:
    kind, payload = _model_state(model_obj)
    if contract.output_kind is None:
        raise ValueError("scoring bundle: в model нужен output_kind (proba | predict)")
    if kind == "lightgbm" and contract.cat_as != "category":
        raise ValueError("scoring bundle: для LightGBM категориальные фичи — cat_as: category")
    state = {
        "model_kind": kind,
        "model": payload,
        "features": list(contract.features),
        "cat_features": list(contract.cat_features),
        "num_dtype": contract.num_dtype,
        "cat_as": contract.cat_as,
        "score_column": contract.score_column,
        "output_kind": contract.output_kind,
        "source_mrid": list(contract.mrid),
    }
    return _materialize(state)


def bundle_from_file(model_path: Union[str, Path], contract: Model) -> Any:
    model_obj = runtime.sk_load_model(str(model_path))
    return build_bundle(model_obj, contract)
