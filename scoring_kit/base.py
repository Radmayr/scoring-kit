class BasePredictor:
    """Локальная замена BasePredictor из airflow_provider_inference.

    Нужна только чтобы predictor.py импортировался и отлаживался на машине без
    провайдеров Airflow. В сгенерированном DAG этот импорт вырезается, и класс
    наследуется от настоящего BasePredictor.
    """

    def setup(self, model_paths: list):
        pass

    def predict(self, df):
        raise NotImplementedError
