import ast
import textwrap

import pandas as pd
import pytest

from scoring_kit.predictor_source import build_predictor_class
from scoring_kit.spec import Model, PipelineError
from scoring_kit.stubs import BasePredictor

MODEL = Model(
    mrid=["t/m/0.0.1"],
    image="img",
    features=["a", "c"],
    cat_features=["c"],
)

SOURCE = textwrap.dedent(
    '''\
    """Модульный docstring."""
    import math
    from collections import (
        OrderedDict,
    )

    from scoring_kit import BasePredictor


    class Predictor(BasePredictor):
        """Docstring класса."""

        threshold = 0.5

        def setup(self, model_paths):
            # комментарий в setup сохраняется
            self.k = math.sqrt(4)

        def predict(self, df):
            """Docstring predict."""
            df["score"] = (df["a"] / (df["a"].max() + 1)) * self.k / 2
            df["cnt"] = len(OrderedDict(a=1))
            return df

        def helper(self):
            return math.pi
    '''
)


def build(source=SOURCE, model=MODEL):
    return build_predictor_class(source, model, "gen_predictor")


def load(code):
    ns = {"BasePredictor": BasePredictor}
    exec(code, ns)
    return ns["gen_predictor"]


def test_structure_and_comments_preserved():
    code = build()
    assert "class gen_predictor(BasePredictor):" in code
    assert "# комментарий в setup сохраняется" in code
    assert "def _user_predict(self, df):" in code
    assert code.count("def predict(self, df):") == 1
    assert "from scoring_kit import BasePredictor" not in code
    # импорт модуля продублирован в каждый пользовательский метод
    assert code.count("import math") == 3
    tree = ast.parse(code)
    cls = tree.body[0]
    names = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
    assert names == ["setup", "_user_predict", "helper", "predict"]


def test_generated_class_runs_without_module_globals():
    cls = load(build())
    p = cls()
    p.setup(["x"])
    df = pd.DataFrame({"a": [1, 2, 3], "c": [1.0, None, 2.0], "key": [1, 2, 3]})
    out = p.predict(df)
    assert list(out["key"]) == [1, 2, 3]
    assert out["score"].between(0, 1).all()
    assert p.helper() == pytest.approx(3.14159, rel=1e-4)
    assert cls.features == ["a", "c"]
    assert cls.cat_features == ["c"]


def test_contract_applied_before_user_predict():
    seen = {}

    source = textwrap.dedent(
        """\
        from scoring_kit import BasePredictor

        class Predictor(BasePredictor):
            def setup(self, model_paths):
                pass

            def predict(self, df):
                self.dtypes = dict(df.dtypes.astype(str))
                df["score"] = 0.5
                return df
        """
    )
    p = load(build(source))()
    p.predict(pd.DataFrame({"a": ["1", "2"], "c": [1.0, 2.0]}))
    seen.update(p.dtypes)
    assert seen == {"a": "float64", "c": "category"}


def test_wrapper_checks_scores():
    source = textwrap.dedent(
        """\
        from scoring_kit import BasePredictor

        class Predictor(BasePredictor):
            def setup(self, model_paths):
                pass

            def predict(self, df):
                df["score"] = 5.0
                return df
        """
    )
    p = load(build(source))()
    with pytest.raises(ValueError, match="вне"):
        p.predict(pd.DataFrame({"a": [1], "c": [1]}))


@pytest.mark.parametrize(
    "source, message",
    [
        ("X = 1\nclass Predictor(BasePredictor):\n    def setup(self, m):\n        pass\n    def predict(self, df):\n        return df\n", "атрибутами класса"),
        ("def f():\n    pass\n", "допускаются только import"),
        ("class Other(BasePredictor):\n    pass\n", "допускаются только import"),
        ("class Predictor:\n    def setup(self, m):\n        pass\n    def predict(self, df):\n        return df\n", "BasePredictor"),
        ("class Predictor(BasePredictor):\n    def setup(self, m):\n        pass\n", "нет метода predict"),
        ("class Predictor(BasePredictor):\n    features = []\n    def setup(self, m):\n        pass\n    def predict(self, df):\n        return df\n", "зарезервированы"),
        ("class Predictor(BasePredictor):\n    def setup(self, m): pass\n    def predict(self, df):\n        return df\n", "с новой строки"),
        ("from x import BasePredictor, y\n", "отдельной строкой"),
        ("class Predictor(BasePredictor)\n", "синтаксическая"),
    ],
)
def test_rejected_sources(source, message):
    with pytest.raises(PipelineError, match=message):
        build(source)
