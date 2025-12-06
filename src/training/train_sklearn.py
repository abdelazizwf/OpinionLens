import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils import load_data

if __name__ == "__main__":
    X_train, _, X_test, y_train, _, y_test = load_data()
    
    with mlflow.start_run(run_name="sklearn-log_reg-basic") as run:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        metrics = {
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_precision": precision_score(y_test, predictions, zero_division=np.nan),
            "test_recall": recall_score(y_test, predictions, zero_division=np.nan),
            "test_f1_score": f1_score(y_test, predictions, zero_division=np.nan),
            "test_roc_auc": roc_auc_score(y_test, predictions),
        }
        
        mlflow.log_metrics(metrics)
        
        con_matrix_fig = ConfusionMatrixDisplay.from_predictions(y_test, predictions).figure_
        roc_fig = RocCurveDisplay.from_predictions(y_test, predictions).figure_
        
        mlflow.log_figure(con_matrix_fig, "figures/confusion_matrix.png")
        mlflow.log_figure(roc_fig, "figures/roc_curve.png")