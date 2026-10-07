"""Встроенные классы предикторов для BatchInferenceOperator.

Оператор берёт исходник класса, вырезает подстроку «BasePredictor» и переименовывает класс в
Predictor, поэтому весь код здесь самодостаточен: хелперы вложены в методы, имён модуля нет.
"""

from __future__ import annotations

import textwrap

from scoring_kit import runtime
from scoring_kit.codegen import helpers, lit
from scoring_kit.spec import Calibrate, Model


def _method(name: str, args: str, helper_funcs: tuple, body: str) -> str:
    inner = (helpers(*helper_funcs) + "\n" if helper_funcs else "") + textwrap.dedent(body).strip() + "\n"
    return textwrap.indent(f"def {name}({args}):\n" + textwrap.indent(inner, "    "), "    ")


def contract(model: Model) -> dict:
    return {
        "name": model.name or "model",
        "score_column": model.score_column,
        "output_kind": model.output_kind,
        "features": list(model.features),
        "cat_features": list(model.cat_features),
        "num_dtype": model.num_dtype,
        "cat_as": model.cat_as,
        "score_range": tuple(model.score_range) if model.score_range is not None else None,
    }


def builtin_predictor_class(models: list[Model], class_name: str) -> str:
    """Предиктор без пользовательского кода: модели в порядке mrid, по скору на модель."""
    contracts = [contract(m) for m in models]
    header = (
        f"class {class_name}(BasePredictor):\n"
        "    # --- scoring-kit: встроенный предиктор; контракт моделей из pipeline.yaml ---\n"
        f"    contracts = {lit(contracts, 4)}\n\n"
    )
    setup = _method(
        "setup",
        "self, model_paths",
        (runtime.sk_load_model,),
        """
        if len(model_paths) != len(self.contracts):
            raise ValueError(f"[scoring-kit] ожидалось {len(self.contracts)} моделей, пришло {len(model_paths)}: {model_paths}")
        self.loaded = []
        for contract, path in zip(self.contracts, model_paths):
            print(f"[scoring-kit] модель {contract['name']}: {path}")
            self.loaded.append(sk_load_model(path))
        """,
    )
    predict = _method(
        "predict",
        "self, df",
        (runtime.sk_prepare_features, runtime.sk_model_scores, runtime.sk_check_scores),
        """
        n_rows = len(df)
        for contract, model in zip(self.contracts, self.loaded):
            X = sk_prepare_features(
                df, contract["features"], contract["cat_features"], contract["num_dtype"], contract["cat_as"]
            )[contract["features"]]
            df[contract["score_column"]] = sk_model_scores(model, X, contract["output_kind"])
            sk_check_scores(df, n_rows, contract["score_column"], contract["score_range"])
        return df
        """,
    )
    return header + setup + "\n" + predict


def calibration_predictor_class(cfg: Calibrate, dev_path: str, report_path: str, class_name: str) -> str:
    """Обучение на лету: dev читается в setup из смонтированного входа, apply приходит в predict."""
    settings = {
        "method": cfg.method,
        "score": cfg.score,
        "target": cfg.target,
        "output": cfg.output,
        "segments": cfg.segments,
        "cohort_column": cfg.cohort_column,
        "train_offsets": list(cfg.train_offsets),
        "uncalibrated": cfg.uncalibrated,
    }
    header = (
        f"class {class_name}(BasePredictor):\n"
        "    # --- scoring-kit: калибровка; настройки из pipeline.yaml ---\n"
        f"    settings = {lit(settings, 4)}\n\n"
    )
    setup = _method(
        "setup",
        "self, model_paths",
        (runtime.sk_read_frame,),
        f"""
        self.dev = sk_read_frame({dev_path!r})
        print(f"[scoring-kit] dev: {{len(self.dev)}} строк")
        """,
    )
    predict = _method(
        "predict",
        "self, df",
        (runtime.sk_calibrate, runtime.sk_write_frame),
        f"""
        result, report = sk_calibrate(self.dev, df, self.settings)
        sk_write_frame(report, {report_path!r})
        return result
        """,
    )
    return header + setup + "\n" + predict
