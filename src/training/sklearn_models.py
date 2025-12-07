from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

conf = OmegaConf.load("params.yaml")


class LogisticRegressionSubject:
    mlflow_run_name = "sklearn-log_reg-tuning"
    
    @classmethod
    def get_params(cls, trial):
        params = {
            "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        }
        return params
    
    @classmethod
    def get_model(cls, params):
        return LogisticRegression(**params, random_state=conf.base.random_seed)


class LinearSVCSubject:
    mlflow_run_name = "sklearn-lin_svc-tuning"
    
    @classmethod
    def get_params(cls, trial):
        params = {
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        }
        return params
    
    @classmethod
    def get_model(cls, params):
        return LinearSVC(**params, random_state=conf.base.random_seed)
