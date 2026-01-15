import mlflow

from opinionlens.common.settings import get_settings

settings = get_settings()

mlflow.set_tracking_uri(
    settings.mlflow.local_tracking_uri
)
mlflow.set_experiment(
    settings.mlflow.local_experiment_name
)
