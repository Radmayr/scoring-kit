"""Превращает predictor.py в самодостаточный класс для BatchInferenceOperator.

В под передаётся только исходник класса, поэтому:
- импорты уровня модуля копируются в начало каждого метода;
- метод predict пользователя переименовывается в _user_predict, а сгенерированный
  predict приводит типы фичей по контракту, вызывает его и проверяет результат;
- параметры контракта (features, cat_features, ...) становятся атрибутами класса.
Правка текстовая (по номерам строк из ast), чтобы сохранились комментарии.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap

from scoring_kit import runtime
from scoring_kit.codefmt import lit
from scoring_kit.spec import Model, PipelineError

CONTRACT_ATTRS = ("features", "cat_features", "num_dtype", "cat_as", "score_column", "score_range")
RESERVED = set(CONTRACT_ATTRS) | {"_user_predict"}


def _err(msg: str) -> PipelineError:
    return PipelineError(f"predictor.py: {msg}")


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _imports_base_predictor(node: ast.stmt) -> bool:
    return isinstance(node, ast.ImportFrom) and any(a.name == "BasePredictor" for a in node.names)


def _assigned_names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    targets = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    return [t.id for t in targets if isinstance(t, ast.Name)]


def _contract_lines(model: Model, indent: str) -> list[str]:
    values = {
        "features": list(model.features),
        "cat_features": list(model.cat_features),
        "num_dtype": model.num_dtype,
        "cat_as": model.cat_as,
        "score_column": model.score_column,
        "score_range": tuple(model.score_range) if model.score_range is not None else None,
    }
    lines = [f"{indent}# --- scoring-kit: контракт модели из pipeline.yaml ---"]
    lines += [f"{indent}{name} = {lit(value, len(indent))}" for name, value in values.items()]
    return lines


def _wrapper_lines(indent: str) -> list[str]:
    helpers = "\n".join(
        inspect.getsource(f) for f in (runtime.sk_prepare_features, runtime.sk_check_scores)
    )
    body = (
        "# scoring-kit: типы фичей по контракту -> ваш predict -> проверка результата\n"
        + helpers
        + "\n"
        "n_rows = len(df)\n"
        "prepared = sk_prepare_features(df, self.features, self.cat_features, self.num_dtype, self.cat_as)\n"
        "result = self._user_predict(prepared)\n"
        "sk_check_scores(result, n_rows, self.score_column, self.score_range)\n"
        "return result\n"
    )
    method = "def predict(self, df):\n" + textwrap.indent(body, "    ")
    return [""] + textwrap.indent(method, indent).splitlines()


def build_predictor_class(source: str, model: Model, new_name: str) -> str:
    """Возвращает исходник класса new_name для вставки в DAG (с отступом 0)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise _err(f"синтаксическая ошибка: {e}") from e

    class_name = model.predictor_class
    imports: list[str] = []
    cls = None
    for i, node in enumerate(tree.body):
        if i == 0 and _is_docstring(node):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if _imports_base_predictor(node):
            if len(node.names) > 1:
                raise _err("импортируйте BasePredictor отдельной строкой")
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(source, node))
            continue
        if isinstance(node, ast.ClassDef) and node.name == class_name and cls is None:
            cls = node
            continue
        raise _err(
            f"строка {node.lineno}: на верхнем уровне допускаются только import и класс "
            f"{class_name}. В под передаётся только тело класса — константы сделайте "
            f"атрибутами класса, функции — методами."
        )
    if cls is None:
        raise _err(f"не найден класс {class_name}")
    if cls.decorator_list:
        raise _err(f"у класса {class_name} не должно быть декораторов")
    if len(cls.bases) != 1 or not (isinstance(cls.bases[0], ast.Name) and cls.bases[0].id == "BasePredictor"):
        raise _err(f"класс {class_name} должен наследоваться ровно от BasePredictor")

    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    for required in ("setup", "predict"):
        if required not in methods:
            raise _err(f"в классе {class_name} нет метода {required}")
    clash = sorted({name for n in cls.body for name in _assigned_names(n)} & RESERVED)
    if clash:
        raise _err(f"имена {clash} зарезервированы scoring-kit — переименуйте")

    lines = source.splitlines()
    first = cls.lineno - 1
    cls_lines = lines[first : cls.end_lineno]

    # (индекс строки в cls_lines, вставляемые строки); применяем снизу вверх.
    inserts: list[tuple[int, list[str]]] = []
    for fn in methods.values():
        body = fn.body
        anchor = body[1] if _is_docstring(body[0]) and len(body) > 1 else body[0]
        if anchor.lineno == fn.lineno:
            raise _err(f"метод {fn.name}: тело должно начинаться с новой строки")
        if imports:
            indent = " " * anchor.col_offset
            at = anchor.lineno - 1 - first
            if anchor is body[0] and _is_docstring(anchor):
                at = anchor.end_lineno - first
            inserts.append((at, textwrap.indent("\n".join(imports), indent).splitlines()))

    body0 = cls.body[0]
    unit = " " * body0.col_offset
    if _is_docstring(body0):
        contract_at = body0.end_lineno - first
    else:
        decorators = getattr(body0, "decorator_list", [])
        contract_at = min([body0.lineno] + [d.lineno for d in decorators]) - 1 - first
    lead = [""] if _is_docstring(body0) else []
    next_line = cls_lines[contract_at] if contract_at < len(cls_lines) else ""
    trail = [""] if next_line.strip() else []
    inserts.append((contract_at, lead + _contract_lines(model, unit) + trail))

    for at, new in sorted(inserts, key=lambda x: x[0], reverse=True):
        cls_lines[at:at] = new

    predict_line = methods["predict"].lineno - 1 - first
    # predict_line сдвинулся на число вставленных выше строк.
    shift = sum(len(new) for at, new in inserts if at <= predict_line)
    idx = predict_line + shift
    cls_lines[idx], n = re.subn(r"\bdef\s+predict\b", "def _user_predict", cls_lines[idx], count=1)
    if n != 1:
        raise _err("не удалось переименовать predict")

    cls_lines[0], n = re.subn(rf"\bclass\s+{re.escape(class_name)}\b", f"class {new_name}", cls_lines[0], count=1)
    if n != 1:
        raise _err("не удалось переименовать класс")

    cls_lines += _wrapper_lines(unit)
    code = "\n".join(cls_lines) + "\n"
    ast.parse(code)  # сгенерированное должно оставаться валидным Python
    return code
