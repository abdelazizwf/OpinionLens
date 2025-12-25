import os

import dotenv
import mlflow

from opinionlens.common.utils import get_logger

logger = get_logger("opinionlens.api")

dotenv.load_dotenv(".env", override=True)

mlflow.set_tracking_uri(
    os.environ["MLFLOW_TRACKING_URI"]
)
logger.info(f"Mlflow tracking URI set as {mlflow.get_tracking_uri()}")

experiment = mlflow.set_experiment(
    os.environ["MLFLOW_EXPERIMENT_NAME"]
)
logger.info(f"Mlflow experiment set as {experiment.name}")
