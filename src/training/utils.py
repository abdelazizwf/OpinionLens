from datetime import datetime

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_vectorized_data():
    X_train = joblib.load("data/vectorized/train_vectors.pkl")
    X_val = joblib.load("data/vectorized/val_vectors.pkl")
    X_test = joblib.load("data/vectorized/test_vectors.pkl")
    
    y_train = joblib.load("data/vectorized/train_scores.pkl")
    y_val = joblib.load("data/vectorized/val_scores.pkl")
    y_test = joblib.load("data/vectorized/test_scores.pkl")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_metrics(y_test, predictions, prefix="", figures=False):
    assert len(y_test) == len(predictions)
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_test, predictions),
        f"{prefix}precision": precision_score(y_test, predictions, zero_division=np.nan),
        f"{prefix}recall": recall_score(y_test, predictions, zero_division=np.nan),
        f"{prefix}f1_score": f1_score(y_test, predictions, zero_division=np.nan),
        f"{prefix}roc_auc": roc_auc_score(y_test, predictions),
    }
    
    if figures:
        con_matrix_fig = ConfusionMatrixDisplay.from_predictions(y_test, predictions).figure_
        roc_fig = RocCurveDisplay.from_predictions(y_test, predictions).figure_
        return metrics, con_matrix_fig, roc_fig
    else:
        return metrics


def concat_data(vectors_list, scores_list):
    assert type(vectors_list) is list and type(scores_list) is list
    vectors = vstack(vectors_list, format="csr")
    scores = np.concat(scores_list)
    return vectors, scores


def get_timestamp():
    time = datetime.now().replace(microsecond=0).isoformat()
    replacements = str.maketrans("", "", "T:-")
    time = time.translate(replacements)
    return time
