from omegaconf import OmegaConf
from optuna import Trial
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

conf = OmegaConf.load("params.yaml")
random_state = conf.base.random_seed
n_jobs = conf.training.n_jobs


class LogisticRegressionSubject:
    mlflow_run_name = "sklearn-log_reg-tuning"
    
    @classmethod
    def get_params(cls, trial: Trial) -> dict:
        params = {
            "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        }
        return params
    
    @classmethod
    def get_model(cls, params: dict) -> LogisticRegression:
        return LogisticRegression(**params, random_state=random_state)


class LinearSVCSubject:
    mlflow_run_name = "sklearn-lin_svc-tuning"
    
    @classmethod
    def get_params(cls, trial: Trial) -> dict:
        params = {
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        }
        return params
    
    @classmethod
    def get_model(cls, params: dict) -> LinearSVC:
        return LinearSVC(**params, random_state=random_state)


class KNNSubject:
    mlflow_run_name = "sklearn-knn-tuning"
    
    @classmethod
    def get_params(cls, trial: Trial) -> dict:
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 8),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["cosine", "l1", "l2"]),
            "p": trial.suggest_float("p", 0.5, 3.0),
        }
        return params
    
    @classmethod
    def get_model(cls, params: dict) -> KNeighborsClassifier:
        return KNeighborsClassifier(**params, n_jobs=n_jobs)


class DecisionTreeSubject:
    mlflow_run_name = "sklearn_decision-tree_tuning"
    
    @classmethod
    def get_params(cls, trial: Trial) -> dict:
        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "log_loss", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 12, 22),
        }
        return params
    
    @classmethod
    def get_model(cls, params: dict) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**params, random_state=random_state)


class BaggingLinearSVCSubject:
    mlflow_run_name = "sklearn-bagging_lin_svc-tuning"
    
    @classmethod
    def get_params(cls, trial: Trial) -> dict:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 5, 20),
            "max_samples": trial.suggest_float("max_samples", 0.25, 1.0),
        }
        return params
    
    @classmethod
    def get_model(cls, params: dict) -> BaggingClassifier:
        return BaggingClassifier(
            **params,
            estimator=LinearSVC(penalty="l2", C=0.4),
            random_state=random_state,
        )


class RandomForestSubject:
    mlflow_run_name = "sklearn-random_forest-tuning"
    
    @classmethod
    def get_params(cls, trial: Trial) -> dict:
        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "log_loss", "entropy"]),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        }
        return params
    
    @classmethod
    def get_model(cls, params: dict) -> RandomForestClassifier:
        return RandomForestClassifier(**params, n_jobs=n_jobs, random_state=random_state)
