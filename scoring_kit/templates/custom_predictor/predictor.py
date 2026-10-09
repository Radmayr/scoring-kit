import joblib

from scoring_kit import BasePredictor  # в DAG заменится на airflow_provider_inference


class Predictor(BasePredictor):
    """Свой предиктор — только если не хватает встроенного (model.output_kind)."""

    def setup(self, model_paths: list[str]):
        # model_paths — пути к артефактам из model.mrid в том же порядке
        self.model = joblib.load(model_paths[0])

    def predict(self, df):
        # Фичи в df уже приведены к типам контракта; порядок колонок модели — self.features.
        df["score"] = self.model.predict_proba(df[self.features])[:, 1]
        return df
