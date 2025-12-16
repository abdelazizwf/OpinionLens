import os
from datetime import datetime

import mlflow

from opinionlens.api.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.api.models import Model, SklearnModel
from opinionlens.common.utils import get_logger

__all__ = ["model_manager"]

SAVED_MODEL_PATH = os.environ["API_SAVED_MODEL_PATH"]


class __ModelManager:
    
    def __init__(self):
        self._hot_models = {}
        self._hot_model_chance = {}
        self._cold_models = set(self._list_model_path_dirs())
        self._default_model_id = None
        self._logger = get_logger(self.__class__.__name__, level=10)
        
        if len(self._cold_models) > 0:
            model_id = self._cold_models.pop()
            self._cold_models.add(model_id)
            self.warm_model(model_id)
            self.set_default(model_id)
        
        self._logger.info("Model manager initialized.")
    
    def _get_model_path(self, model_id: str) -> str:
        return os.path.join(SAVED_MODEL_PATH, model_id)
    
    def _download_model(self, model_uri: str, model_id: str) -> str:
        self._logger.info(
            f"Model {model_id!r} was requested with URI {model_uri!r} from the registry."
        )
        
        dst_path = self._get_model_path(model_id)
        
        if not os.path.exists(dst_path):
            dst_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=dst_path,
            )
            self._logger.debug(f"Model {model_id!r} downloaded and saved at {dst_path!r}.")
            self._cold_models.add(model_id)
            self._logger.debug(f"Model {model_id!r} added to cold models.")
        else:
            self._logger.debug(f"Model {model_id!r} found at {dst_path!r}.")
        
        return dst_path
    
    def _list_model_path_dirs(self) -> list[str]:
        _, dirs, _ = next(iter(os.walk(SAVED_MODEL_PATH)))
        return dirs
    
    def _model_is_hot(self, model_id: str) -> bool:
        return model_id in self._hot_models.keys()
    
    def _model_is_cold(self, model_id: str) -> bool:
        return model_id in self._cold_models
    
    def get_default_model(self) -> Model:
        if self._default_model_id is None:
            raise ModelNotAvailableError("No default model set.")
        
        if not self._model_is_hot(self._default_model_id):
            raise OperationalError(f"Model {self._default_model_id!r} was requested but isn't hot.")
        
        self._logger.info(f"Model {self._default_model_id!r} was requested.")
        
        return self._hot_models[self._default_model_id]
    
    def get_sampled_model(self):
        raise NotImplementedError()
    
    def fetch_model_by_id(self, model_id: str) -> str:
        model_uri = f"models:/{model_id}"
        dst_path = self._download_model(model_uri, model_id)
        return dst_path
    
    def fetch_model_by_name(
        self, model_name: str, model_version: int | None = None, model_alias: str | None = None
    ) -> tuple[str, str]:
        assert model_alias is not None or model_version is not None
    
        if model_alias:
            model_uri = f"models:/{model_name}@{model_alias}"
        else:
            model_uri = f"models:/{model_name}/{model_version}"
        
        model_id = mlflow.models.get_model_info(model_uri).model_id
        dst_path = self._download_model(model_uri, model_id)
        
        return dst_path, model_id
    
    def delete_model(self, model_id):
        raise NotImplementedError()
    
    def list_models(self, hot_only: bool = False) -> list[dict]:
        model_ids = self._list_model_path_dirs()

        result = []
        for model_id in model_ids:
            if hot_only and self._model_is_cold(model_id):
                continue
            
            model_info = mlflow.models.get_model_info(
                self._get_model_path(model_id)
            )
            result.append({
                "model_name": model_info.name,
                "model_id": model_info.model_id,
                "is_hot": self._model_is_hot(model_info.model_id),
                "is_default": True if self._default_model_id == model_info.model_id else False,
                "model_creation": datetime.fromtimestamp(model_info.creation_timestamp / 1000).replace(microsecond=0),
                "model_flavors": list(model_info.flavors.keys()),
                "model_tags": model_info.tags,
            })
        
        return result
    
    def warm_model(self, model_id: str):
        if self._model_is_hot(model_id):
            self._logger.info(f"Model {model_id!r} is hot.")
            return
        
        model_path = self._get_model_path(model_id)
        model = SklearnModel(model_id, model_path)
        
        self._hot_models[model_id] = model
        self._cold_models.remove(model_id)
        
        self._logger.info(f"Model {model_id!r} warmed.")
    
    def cool_model(self, model_id: str):
        if self._model_is_cold(model_id):
            self._logger.info(f"Model {model_id!r} is cold.")
            return
        
        del self._hot_models[model_id]
        self._cold_models.add(model_id)
        
        self._logger.info(f"Model {model_id!r} cooled.")
    
    def set_default(self, model_id: str):
        if not self._model_is_hot(model_id):
            raise OperationalError(f"Model {model_id!r} is not hot and can't be the default.")
        
        self._default_model_id = model_id
        
        self._logger.info(f"Model {model_id!r} set as default.")


model_manager = __ModelManager()
