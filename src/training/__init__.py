import os

import dotenv
import mlflow

dotenv.load_dotenv(".env.dev", override=True)

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI")
)
mlflow.set_experiment(
    os.environ.get("MLFLOW_EXPERIMENT_NAME")
)
