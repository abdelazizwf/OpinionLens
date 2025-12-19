import json
import os
import shutil
from datetime import datetime
from typing import Any

import mlflow

from opinionlens.api.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.api.models import Model, SklearnModel
from opinionlens.common.utils import get_logger

__all__ = ["model_manager"]

SAVED_MODEL_PATH = os.environ["API_SAVED_MODEL_PATH"]


class __ModelManager:
    
    def __init__(self):
        self._models = {}
        self._model_infos = {}
        self._model_chance = {}
        self._default_model_id = None
        self._logger = get_logger(self.__class__.__name__, level=10)
        
        for model_id in set(self._list_model_path_dirs()):
            self.fetch_model("models:/" + model_id)
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
        else:
            self._logger.debug(f"Model {model_id!r} found at {dst_path!r}.")
        
        return dst_path
    
    def _list_model_path_dirs(self) -> list[str]:
        _, dirs, _ = next(iter(os.walk(SAVED_MODEL_PATH)))
        return dirs
    
    def _model_exists(self, model_id: str) -> bool:
        return model_id in self._models.keys()
    
    def _remove_model_dir(self, model_id: str):
        model_path = self._get_model_path(model_id)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            self._logger.debug(f"Deleted model directory {model_path!r}.")
        else:
            raise OperationalError(f"Model directory {model_path!r} doesn't exist.")
    
    def _load_model(self, model_id: str):
        if self._model_exists(model_id):
            self._logger.info(f"Model {model_id!r} is already loaded.")
            return
        
        model_path = self._get_model_path(model_id)
        model = SklearnModel(model_id, model_path)
        
        self._models[model_id] = model
        
        self._logger.info(f"Model {model_id!r} loaded.")
    
    def _format_model_info(
        self,
        model_info: "mlflow.models.model.ModelInfo",
        model_uri: str,
        dst_path: str
    ) -> dict[str, Any]:
        result = {
            "model_name": model_info.name,
            "model_id": model_info.model_id,
            "is_default": True if self._default_model_id == model_info.model_id else False,
            "model_creation": datetime.fromtimestamp(model_info.creation_timestamp / 1000).replace(microsecond=0),
            "model_flavors": list(model_info.flavors.keys()),
            "registry_model_uri": model_uri,
            "saved_model_path": dst_path,
        }
        
        versions = model_info.tags["mlflow.modelVersions"]
        versions = json.loads(model_info.tags["mlflow.modelVersions"]) if versions is not None else None
        model_info.tags["mlflow.modelVersions"] = versions
        result["model_tags"] = model_info.tags
        
        return result
    
    def get_default_model(self) -> Model:
        if self._default_model_id is None:
            raise ModelNotAvailableError("No default model set.")
        
        if not self._model_exists(self._default_model_id):
            raise OperationalError(f"Model {self._default_model_id!r} was requested but doesn't exist.")
        
        self._logger.info(f"Model {self._default_model_id!r} was requested.")
        
        return self._models[self._default_model_id]
    
    def get_sampled_model(self):
        raise NotImplementedError()
    
    def fetch_model(self, model_uri: str) -> tuple[str, str]:
        model_info = mlflow.models.get_model_info(model_uri)
        model_id = model_info.model_id
        
        if self._model_exists(model_id):
            self._logger.info(f"Model {model_id!r}, requested as {model_uri!r}, is already loaded.")
            dst_path = self._get_model_path(model_id)
            return dst_path, model_id
        
        dst_path = self._download_model(model_uri, model_id)        
        self._load_model(model_id)
        self._model_infos[model_id] = self._format_model_info(model_info, model_uri, dst_path)
        
        return dst_path, model_id
    
    def list_models(self) -> list[dict]:
        return list(self._model_infos.values())
    
    def delete_model(self, model_id: str):
        if not self._model_exists(model_id):
            raise ModelNotAvailableError(f"Model {model_id!r} doesn't exist.")
        
        del self._models[model_id]
        self._remove_model_dir(model_id)
        del self._model_infos[model_id]
        
        if self._default_model_id == model_id:
            self._default_model_id = None
        
        self._logger.info(f"Model {model_id!r} deleted.")
    
    def set_default(self, model_id: str):
        if not self._model_exists(model_id):
            raise OperationalError(f"Model {model_id!r} doesn't exist and can't be the default.")
        
        if self._default_model_id is not None:
            self._model_infos[self._default_model_id]["is_default"] = False
        
        self._default_model_id = model_id
        self._model_infos[model_id]["is_default"] = True
        
        self._logger.info(f"Model {model_id!r} set as default.")


model_manager = __ModelManager()
