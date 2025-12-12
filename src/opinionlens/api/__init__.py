import os

import dotenv
import mlflow

dotenv.load_dotenv(".env.dev", override=True)

mlflow.set_tracking_uri(
    os.environ["MLFLOW_TRACKING_URI"]
)
mlflow.set_experiment(
    "Default"
)
