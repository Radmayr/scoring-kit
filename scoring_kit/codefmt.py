"""Форматирование Python-литералов для генерируемого кода."""


def lit(value, indent: int = 0) -> str:
    """Python-литерал для строки кода с отступом indent.

    Многострочные строки — в тройных кавычках, длинные dict/list — по элементу на строку.
    """
    if (
        isinstance(value, str)
        and "\n" in value
        and "'''" not in value
        and "\\" not in value
        and not value.endswith("'")
    ):
        return "'''" + value + "'''"
    one_line = repr(value)
    if not isinstance(value, (dict, list)) or indent + len(one_line) <= 96:
        return one_line
    pad = " " * (indent + 4)
    if isinstance(value, dict):
        items = [f"{pad}{k!r}: {lit(v, indent + 4)}," for k, v in value.items()]
        return "{\n" + "\n".join(items) + "\n" + " " * indent + "}"
    items = [f"{pad}{lit(v, indent + 4)}," for v in value]
    return "[\n" + "\n".join(items) + "\n" + " " * indent + "]"
