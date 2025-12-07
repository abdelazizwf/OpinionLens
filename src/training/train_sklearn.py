import mlflow
from sklearn.linear_model import LogisticRegression

from .utils import calculate_metrics, concat_data, load_vectorized_data

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_vectorized_data()
    
    with mlflow.start_run(run_name="sklearn-log_reg-basic"):
        model = LogisticRegression()
        train_vectors, train_scores = concat_data([X_train, X_val], [y_train, y_val])
        model.fit(train_vectors, train_scores)
        predictions = model.predict(X_test)
        
        metrics, con_matrix_fig, roc_fig = calculate_metrics(
            y_test, predictions, prefix="test_", figures=True
        )
        
        mlflow.log_metrics(metrics)
        
        mlflow.log_figure(con_matrix_fig, "figures/confusion_matrix.png")
        mlflow.log_figure(roc_fig, "figures/roc_curve.png")
