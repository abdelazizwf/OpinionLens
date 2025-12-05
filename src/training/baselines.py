import random
from collections import Counter

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def zero_rule_baseline(train_data: pd.DataFrame, test_data: pd.DataFrame):
    scores = train_data["score"]
    most_common_score = Counter(scores).most_common(1)[0][0]
    predictions = [most_common_score] * len(test_data)
    return predictions

def random_baseline(train_data: pd.DataFrame, test_data: pd.DataFrame):
    return [random.randint(0, 1) for _ in range(len(test_data))]


def heuristic_baseline(train_data: pd.DataFrame, test_data: pd.DataFrame):
    pass


if __name__ == "__main__":
    train_data = pd.read_csv("./data/preprocessed/imdb_dataset/train.csv")
    test_data = pd.read_csv("./data/preprocessed/imdb_dataset/test.csv")
    
    for func in [
        zero_rule_baseline, random_baseline,
    ]:
        with mlflow.start_run(run_name=func.__name__) as run:
            predictions = func(train_data, test_data)
            truths = test_data["score"].to_list()
            assert len(truths) == len(predictions), f"{len(truths)} != {len(predictions)}"
            
            metrics = {
                "test_accuracy": accuracy_score(truths, predictions),
                "test_precision": precision_score(truths, predictions, zero_division=np.nan),
                "test_recall": recall_score(truths, predictions, zero_division=np.nan),
                "test_f1_score": f1_score(truths, predictions, zero_division=np.nan),
                "test_roc_auc": roc_auc_score(truths, predictions),
            }
            
            mlflow.log_metrics(metrics)
            
            con_matrix_fig = ConfusionMatrixDisplay.from_predictions(truths, predictions).figure_
            roc_fig = RocCurveDisplay.from_predictions(truths, predictions).figure_
            
            mlflow.log_figure(con_matrix_fig, "figures/confusion_matrix.png")
            mlflow.log_figure(roc_fig, "figures/roc_curve.png")
