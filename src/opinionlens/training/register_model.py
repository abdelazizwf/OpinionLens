import os
import sys

import mlflow


def main():
    model_id = sys.argv[1]
    model_name = sys.argv[2]

    model_uri = f"models:/{model_id}"
    model = mlflow.sklearn.load_model(model_uri)
    model_info = mlflow.models.get_model_info(model_uri)

    remote_uri = os.environ["MLFLOW_REMOTE_URI"]

    mlflow.set_tracking_uri(remote_uri)
    mlflow.set_experiment("Default")

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            model,
            registered_model_name=model_name,
            signature=model_info.signature,
            params=model_info.params,
            metadata=model_info.metadata,
            name=model_info.name,
            tags=model_info.tags,
        )


if __name__ == "__main__":
    main()
