"""Сборка текста DAG'а: импорты, модульные блоки (колбэки, классы), таски, связи.

Блоки (scoring_kit.blocks) добавляют сюда код тасков; рецепты (scoring_kit.recipes) решают,
какие блоки и в каком порядке. Здесь — только механика: отступы, литералы, порядок секций.
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Callable

from scoring_kit.codefmt import lit

IND = 4  # отступ тела `with DAG(...)`


def helpers(*funcs: Callable) -> str:
    return "\n".join(inspect.getsource(f) for f in funcs)


def func(name: str, args: str, helper_funcs: tuple, body: str) -> str:
    """Функция уровня модуля с вложенными самодостаточными хелперами."""
    inner = (helpers(*helper_funcs) + "\n" if helper_funcs else "") + textwrap.dedent(body).strip() + "\n"
    return f"def {name}({args}):\n" + textwrap.indent(inner, "    ")


def call(target: str, cls: str, kwargs: dict, indent: int) -> str:
    """target = Cls(k=v, ...); значения — готовые строки кода."""
    pad = " " * (indent + 4)
    args = "".join(f"{pad}{k}={v},\n" for k, v in kwargs.items())
    return f"{' ' * indent}{target} = {cls}(\n{args}{' ' * indent})"


class DagBuilder:
    def __init__(self) -> None:
        self.imports: list[str] = []
        self.blocks: list[str] = []
        self.body: list[str] = []
        self.edges: list[tuple[str, str]] = []

    def need(self, *lines: str) -> None:
        for line in lines:
            if line not in self.imports:
                self.imports.append(line)

    def block(self, code: str) -> None:
        self.blocks.append(code.rstrip("\n"))

    def task(self, code: str) -> None:
        self.body.append(code.rstrip("\n"))
        self.body.append("")

    def group(self, name: str, tasks: list[str], chain: list[str]) -> None:
        """TaskGroup с последовательной цепочкой тасков внутри."""
        self.body.append(f'{" " * IND}with TaskGroup("{name}", prefix_group_id=False) as {name}:')
        for code in tasks:
            self.body.append(code.rstrip("\n"))
            self.body.append("")
        self.body.append(" " * (IND + 4) + " >> ".join(chain))
        self.body.append("")

    def edge(self, upstream: str, downstream: str) -> None:
        self.edges.append((upstream, downstream))

    def render(self, header_comment: str, dag_open: list[str]) -> str:
        imports = sorted(self.imports, key=lambda s: (not s.startswith("from datetime"), not s.startswith("import"), s))
        head = header_comment + "\n" + "\n".join(imports)
        body = list(dag_open) + self.body
        # Связи: подряд идущие пары a >> b >> c склеиваем в цепочки для читаемости.
        for line in _chains(self.edges):
            body.append(" " * IND + line)
        parts = [head, *self.blocks, "\n".join(body)]
        code = "\n\n\n".join(parts) + "\n"
        return code


def _chains(edges: list[tuple[str, str]]) -> list[str]:
    lines: list[str] = []
    used = [False] * len(edges)
    for i, (a, b) in enumerate(edges):
        if used[i]:
            continue
        used[i] = True
        chain = [a, b]
        extended = True
        while extended:
            extended = False
            for j, (c, d) in enumerate(edges):
                if not used[j] and c == chain[-1] and sum(1 for x, _ in edges if x == c) == 1:
                    chain.append(d)
                    used[j] = True
                    extended = True
                    break
        lines.append(" >> ".join(chain))
    return lines


__all__ = ["DagBuilder", "IND", "call", "func", "helpers", "lit"]
