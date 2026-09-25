import joblib

from scoring_kit import BasePredictor  # в DAG заменится на airflow_provider_inference


class Predictor(BasePredictor):
    """LightGBM (sklearn API), артефакт сохранён через joblib."""

    def setup(self, model_paths: list[str]):
        # model_paths — пути к артефактам из model.mrid в том же порядке
        print("Пути до моделей:", model_paths)
        self.model = joblib.load(model_paths[0])

    def predict(self, df):
        # Фичи в df уже приведены к типам контракта; порядок колонок модели — self.features.
        df["score"] = self.model.predict_proba(df[self.features])[:, 1]
        return df
