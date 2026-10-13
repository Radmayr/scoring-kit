"""scoring-kit: yaml + predictor.py -> самодостаточный Airflow DAG батч-скоринга."""

from scoring_kit.base import BasePredictor
from scoring_kit.bundle import PackError, load_packed, pack_model

__version__ = "0.7.2"

__all__ = ["BasePredictor", "Pipeline", "load_pipeline", "pack_model", "load_packed", "PackError", "__version__"]


def __getattr__(name):
    # Конфиги тянут pydantic; упаковке модели в ноутбуке он не нужен, поэтому импорт ленивый.
    if name in ("Pipeline", "load_pipeline"):
        from scoring_kit import spec

        return getattr(spec, name)
    raise AttributeError(f"module 'scoring_kit' has no attribute {name!r}")
