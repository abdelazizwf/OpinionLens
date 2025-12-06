import mlflow
from sklearn.linear_model import LogisticRegression

from .utils import calculate_metrics, load_data

if __name__ == "__main__":
    X_train, _, X_test, y_train, _, y_test = load_data()
    
    with mlflow.start_run(run_name="sklearn-log_reg-basic"):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        metrics, con_matrix_fig, roc_fig = calculate_metrics(y_test, predictions, prefix="test_")
        
        mlflow.log_metrics(metrics)
        
        mlflow.log_figure(con_matrix_fig, "figures/confusion_matrix.png")
        mlflow.log_figure(roc_fig, "figures/roc_curve.png")
