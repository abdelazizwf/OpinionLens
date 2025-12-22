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
LOGGING_LEVEL = os.environ["LOGGING_LEVEL"]


class __ModelManager:
    """A class to manage models saved on disk at the backend.
    
    DO NOT INSTANTIATE, use `opinionlens.api.manager.model_manager` instead.
    """
    
    def __init__(self):
        self._models = {}
        self._model_infos = {}
        self._model_chance = {}
        self._default_model_id = None
        self._logger = get_logger(self.__class__.__name__, level=LOGGING_LEVEL)
        
        for model_id in set(self._list_model_path_dirs()):
            self.fetch_model("models:/" + model_id)
            self.set_default(model_id)
        
        self._logger.info("Model manager initialized.")
    
    def _get_model_path(self, model_id: str) -> str:
        """Get the path of the model directory."""
        return os.path.join(SAVED_MODEL_PATH, model_id)
    
    def _download_model(self, model_uri: str, model_id: str) -> str:
        """Download the model from the registry."""
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
        """List all available models on disk."""
        _, dirs, _ = next(iter(os.walk(SAVED_MODEL_PATH)))
        return dirs
    
    def _model_exists(self, model_id: str) -> bool:
        """Check if the model is loaded."""
        return model_id in self._models.keys()
    
    def _remove_model_dir(self, model_id: str):
        """Delete the model directory from disk."""
        model_path = self._get_model_path(model_id)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            self._logger.debug(f"Deleted model directory {model_path!r}.")
        else:
            raise OperationalError(f"Model directory {model_path!r} doesn't exist.")
    
    def _load_model(self, model_id: str):
        """Load a model from disk."""
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
        """Format the model info."""
        result = {
            "model_name": model_info.name,
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
        """Return the default model object to make predictions.
        
        Returns:
            The default model object.
        """
        if self._default_model_id is None:
            raise ModelNotAvailableError("No default model set.")
        
        if not self._model_exists(self._default_model_id):
            raise OperationalError(f"Model {self._default_model_id!r} was requested but doesn't exist.")
        
        self._logger.info(f"Model {self._default_model_id!r} was requested.")
        
        return self._models[self._default_model_id]
    
    def get_sampled_model(self):
        raise NotImplementedError()
    
    def fetch_model(self, model_uri: str) -> tuple[str, str]:
        """Download and load the requested model from the registry.
        
        Args:
            model_uri: The URI of the model in the registry.
                The URI must start with 'models:/'.
        
        Returns:
            The path of the model directory and the model id.
        
        Raises:
            OperationalError: An error occurred downloading the model.
                Check MLflow service status or the provided URI.
        """
        try:
            model_info = mlflow.models.get_model_info(model_uri)
            model_id = model_info.model_id
        except mlflow.exceptions.MlflowException as e:
            raise OperationalError(f"MLflow error: {e.message}")
        
        if self._model_exists(model_id):
            self._logger.info(f"Model {model_id!r}, requested as {model_uri!r}, is already loaded.")
            dst_path = self._get_model_path(model_id)
            return dst_path, model_id
        
        dst_path = self._download_model(model_uri, model_id)        
        self._load_model(model_id)
        self._model_infos[model_id] = self._format_model_info(model_info, model_uri, dst_path)
        
        return dst_path, model_id
    
    def get_model_info(
        self, model_id: str | None = None
    ) -> dict[str, Any] | dict[str, dict[str, Any]]:
        """Return the model information of the given model ID or all loaded models.
        
        Args:
            model_id: The ID of the requested model. 
                If `None`, then all models' information are returned.
        
        Returns:
            A dictionary containing the requested model's information, or
            a dictionary with all loaded model IDs pointing to the model's information.
        """
        if model_id:
            return self._model_infos[model_id]
        else:
            return self._model_infos
    
    def delete_model(self, model_id: str):
        """Delete the model from the backend.
        
        The model is unloaded from memory by deleting its object reference. Then
        the model directory is deleted from disk.
        
        Args:
            model_id: The ID of the model.
        """
        if not self._model_exists(model_id):
            raise ModelNotAvailableError(f"Model {model_id!r} doesn't exist.")
        
        del self._models[model_id]
        self._remove_model_dir(model_id)
        del self._model_infos[model_id]
        
        if self._default_model_id == model_id:
            self._default_model_id = None
        
        self._logger.info(f"Model {model_id!r} deleted.")
    
    def set_default(self, model_id: str):
        """Set the model with the given ID as the default model.
        
        Args:
            model_id: The ID of the model.
        """
        if not self._model_exists(model_id):
            raise OperationalError(f"Model {model_id!r} doesn't exist and can't be the default.")
        
        if self._default_model_id is not None:
            self._model_infos[self._default_model_id]["is_default"] = False
        
        self._default_model_id = model_id
        self._model_infos[model_id]["is_default"] = True
        
        self._logger.info(f"Model {model_id!r} set as default.")


model_manager = __ModelManager()
