import os
import sys

import mlflow
import pandas as pd
from omegaconf import OmegaConf

from opinionlens.common.utils import get_csv_files
from opinionlens.training.utils import calculate_metrics

conf = OmegaConf.load("params.yaml")

def main():
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = conf.models.model_id

    model = mlflow.sklearn.load_model(f"models:/{model_id}")

    eval_data_path = "data/eval_data/"
    files = get_csv_files(eval_data_path)

    with mlflow.start_run(run_name="evals"):
        mlflow.log_param("model_id", model_id)

        for file in files:
            name = os.path.basename(file).split(".")[0]
            data = pd.read_csv(os.path.join(eval_data_path, f"{name}.csv"))

            with mlflow.start_run(nested=True, run_name=name):
                predictions = model.predict(data["text"])

                metrics, con_matrix_fig, roc_fig = calculate_metrics(
                    data["score"], predictions, prefix="test_", figures=True
                )

                mlflow.log_metrics(metrics)
                mlflow.log_figure(con_matrix_fig, "figures/confusion_matrix.png")
                mlflow.log_figure(roc_fig, "figures/roc_curve.png")


if __name__ == "__main__":
    main()
