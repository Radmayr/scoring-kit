"""scoring-kit: yaml + predictor.py -> самодостаточный Airflow DAG батч-скоринга."""

from scoring_kit.base import BasePredictor
from scoring_kit.spec import Pipeline, load_pipeline

__version__ = "0.4.0"

__all__ = ["BasePredictor", "Pipeline", "load_pipeline", "__version__"]
