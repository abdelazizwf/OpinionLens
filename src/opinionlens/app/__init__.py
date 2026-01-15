import mlflow

from opinionlens.common.settings import get_settings
from opinionlens.common.utils import get_logger

settings = get_settings()

logger = get_logger("opinionlens.app")


mlflow.set_tracking_uri(
    settings.mlflow.remote_tracking_uri
)
logger.info(f"Mlflow tracking URI set as {mlflow.get_tracking_uri()}")

experiment = mlflow.set_experiment(
    settings.mlflow.remote_experiment_name
)
logger.info(f"Mlflow experiment set as {experiment.name}")
