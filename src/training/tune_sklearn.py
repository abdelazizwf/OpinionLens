import mlflow
import optuna
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .utils import calculate_metrics, load_data

conf = OmegaConf.load("params.yaml")

X_train, X_val, X_test, y_train, y_val, y_test = load_data()


def log_reg_trial(trial):
    params = {
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
    }
        
    model = LogisticRegression(**params, random_state=conf.base.random_seed)
    
    return params, model


def lin_svc_objective(trial):
    params = {
        "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
    }
    
    model = LinearSVC(**params, random_state=conf.base.random_seed)
    
    return params, model


if __name__ == "__main__":
    with mlflow.start_run(run_name="sklearn-log_reg-tuning") as run:
        study = optuna.create_study(
            study_name=run.info.run_name,
            direction="maximize",
            storage="sqlite:///optuna.sqlite3",
        )
        
        n_trials = conf.training.n_trials
        for _ in range(n_trials):
            trial = study.ask()
            
            params, model = log_reg_trial(trial)
            
            with mlflow.start_run(nested=True) as run:
                mlflow.log_params(params)
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                metrics, _, _ = calculate_metrics(y_val, predictions, prefix="val_")
                mlflow.log_metrics(metrics)
                trial.set_user_attr("run_name", run.info.run_name)
            
            study.tell(trial, metrics["val_accuracy"])
        
        best_trial = study.best_trial
        best_params = best_trial.params
        
        mlflow.log_params(best_params)
        
        model = LogisticRegression(
            **best_params,
            warm_start=True,
            random_state=conf.base.random_seed,
        )
        
        mlflow.log_param("best_run", best_trial.user_attrs["run_name"])
        mlflow.log_param("val_accuracy", best_trial.value)
        
        model.fit(X_train, y_train)
        model.fit(X_val, y_val)
        
        predictions = model.predict(X_test)
        metrics, con_matrix_fig, roc_fig = calculate_metrics(y_test, predictions, prefix="test_")
        mlflow.log_metrics(metrics)
        mlflow.log_figure(con_matrix_fig, "figures/confusion_matrix.png")
        mlflow.log_figure(roc_fig, "figures/roc.png")
