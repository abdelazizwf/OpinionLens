import mlflow
import optuna
from omegaconf import OmegaConf
from sklearn.pipeline import make_pipeline

from opinionlens.common.utils import get_timestamp
from opinionlens.preprocessing.vectorize import get_saved_tfidf_vectorizer
from opinionlens.training.sklearn_subjects import BaggingLinearSVCSubject
from opinionlens.training.utils import (
    calculate_metrics,
    concat_data,
    load_vectorized_data,
)

conf = OmegaConf.load("params.yaml")


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_vectorized_data()

    subject = BaggingLinearSVCSubject
    run_name = subject.mlflow_run_name
    with mlflow.start_run(run_name=run_name) as run:
        study = optuna.create_study(
            study_name=get_timestamp() + "-" + run_name,
            direction="maximize",
            storage="sqlite:///optuna.sqlite3",
        )

        for n in range(conf.training.n_trials):
            trial = study.ask()

            params = subject.get_params(trial)
            model = subject.get_model(params)

            with mlflow.start_run(nested=True, run_name=f"trial_{n + 1}") as run:
                mlflow.log_params(params)
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                metrics = calculate_metrics(y_val, predictions, prefix="val_")
                mlflow.log_metrics(metrics)
                trial.set_user_attr("run_name", run.info.run_name)

            study.tell(trial, metrics["val_accuracy"])

        best_trial = study.best_trial
        best_params = best_trial.params

        mlflow.log_params(best_params)

        model = subject.get_model(best_params)

        mlflow.log_param("best_run", best_trial.user_attrs["run_name"])
        mlflow.log_param("val_accuracy", best_trial.value)

        train_vectors, train_scores = concat_data([X_train, X_val], [y_train, y_val])
        model.fit(train_vectors, train_scores)

        predictions = model.predict(X_test)
        metrics, con_matrix_fig, roc_fig = calculate_metrics(
            y_test, predictions, prefix="test_", figures=True
        )
        mlflow.log_metrics(metrics)
        mlflow.log_figure(con_matrix_fig, "figures/confusion_matrix.png")
        mlflow.log_figure(roc_fig, "figures/roc.png")

        exp_name = mlflow.get_experiment(run.info.experiment_id).name
        model_name = exp_name + "-" + "-".join(run_name.split("-")[:2])

        model = make_pipeline(
            get_saved_tfidf_vectorizer(), model
        )

        model_info = mlflow.sklearn.log_model(
            model,
            name=model_name,
            # input_example=X_test[0],
        )

        conf.models.model_id = model_info.model_id
        OmegaConf.save(conf, "params.yaml")


if __name__ == "__main__":
    main()
