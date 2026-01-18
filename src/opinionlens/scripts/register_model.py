import os
import sys
from getpass import getpass

import mlflow

from opinionlens.common.settings import get_settings

settings = get_settings()


def main():
    model_id = sys.argv[1]
    model_name = sys.argv[2]

    mlflow.set_tracking_uri(settings.mlflow.local_tracking_uri)
    mlflow.set_experiment(settings.mlflow.local_experiment_name)

    model_uri = f"models:/{model_id}"
    model = mlflow.sklearn.load_model(model_uri)
    model_info = mlflow.models.get_model_info(model_uri)

    remote_password = getpass("Remote MLflow Password: ")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = remote_password

    mlflow.set_tracking_uri(settings.mlflow.remote_tracking_uri)
    mlflow.set_experiment(settings.mlflow.remote_experiment_name)

    with mlflow.start_run():
        remote_model_info = mlflow.sklearn.log_model(
            model,
            registered_model_name=model_name,
            # signature=model_info.signature,
            params=model_info.params,
            metadata=model_info.metadata,
            name=model_info.name,
            tags=model_info.tags,
        )

    print(f"Remote model ID: {remote_model_info.model_id}")
    print(f"Remote model name: {remote_model_info.name}")
    print(f"Remote model registration version: {remote_model_info.registered_model_version}")

    if objs := model_info.tags.get("objects"):
        objs = objs.split("&")
        print("> Don't forget to upload the following model-related objects:")
        for obj in objs:
            print(f"\t{obj}")


if __name__ == "__main__":
    main()
