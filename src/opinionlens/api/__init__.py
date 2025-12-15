import os

import dotenv
import mlflow

dotenv.load_dotenv(".env", override=True)

mlflow.set_tracking_uri(
    os.environ["MLFLOW_REMOTE_URI"]
)
mlflow.set_experiment(
    "Default"
)
