import os
from datetime import datetime

import mlflow
import numpy as np
from scipy.sparse import spmatrix

from opinionlens.api.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.common.utils import get_logger
from opinionlens.preprocessing import clean_text, get_saved_tfidf_vectorizer, tokenizer

SAVED_MODEL_PATH = os.environ["API_SAVED_MODEL_PATH"]


class SklearnModel:
    
    def __init__(self, model_id: str, model_path: str):
        self.model_id = model_id
        self.pyfunc_model = mlflow.pyfunc.load_model(model_path)
        self.logger = get_logger(self.__class__.__name__, level=10)
    
    def preprocess_text(self, text: str) -> spmatrix:
        tokenized_text = " ".join(tokenizer(clean_text(text)))

        vectorizer = get_saved_tfidf_vectorizer()
        vectors = vectorizer.transform(np.array([tokenized_text]))
        
        self.logger.debug("Preprocessing done.")
        return vectors
    
    def predict(self, text: str) -> int:
        self.logger.debug(f"Asked to predict {text!r}.")
        vectors = self.preprocess_text(text)
        prediction = int(self.pyfunc_model.predict(vectors)[0])
        self.logger.debug(f"Prediction result is {prediction!r}.")
        return prediction
    
    def batch_predict(self, batch: list[str]) -> list[int]:
        self.logger.debug(f"Asked to bacth predict a list of length {len(batch)!r}.")
        predictions = [self.predict(text) for text in batch]
        return predictions


class ModelManager:
    
    def __init__(self):
        self.hot_models = {}
        self.hot_model_chance = {}
        self.cold_models = set(self._list_model_path_dirs())
        self.default_model_id = None
        self.logger = get_logger(self.__class__.__name__, level=10)
        
        if len(self.cold_models) > 0:
            model_id = self.cold_models.pop()
            self.cold_models.add(model_id)
            self.warm_model(model_id)
            self.set_default(model_id)
        
        self.logger.info("Model manager initialized.")
    
    def _get_model_path(self, model_id: str) -> str:
        return os.path.join(SAVED_MODEL_PATH, model_id)
    
    def _download_model(self, model_uri: str, model_id: str) -> str:
        self.logger.info(
            f"Model {model_id!r} was requested with URI {model_uri!r} from the registry."
        )
        
        dst_path = self._get_model_path(model_id)
        
        if not os.path.exists(dst_path):
            dst_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=dst_path,
            )
            self.logger.debug(f"Model {model_id!r} downloaded and saved at {dst_path!r}.")
            self.cold_models.add(model_id)
            self.logger.debug(f"Model {model_id!r} added to cold models.")
        else:
            self.logger.debug(f"Model {model_id!r} found at {dst_path!r}.")
        
        return dst_path
    
    def _list_model_path_dirs(self) -> list[str]:
        _, dirs, _ = next(iter(os.walk(SAVED_MODEL_PATH)))
        return dirs
    
    def _model_is_hot(self, model_id: str) -> bool:
        return model_id in self.hot_models.keys()
    
    def _model_is_cold(self, model_id: str) -> bool:
        return model_id in self.cold_models
    
    def get_default_model(self) -> SklearnModel:
        if self.default_model_id is None:
            raise ModelNotAvailableError("No default model set.")
        
        if not self._model_is_hot(self.default_model_id):
            raise OperationalError(f"Model {self.default_model_id!r} was requested but isn't hot.")
        
        self.logger.info(f"Model {self.default_model_id!r} was requested.")
        
        return self.hot_models[self.default_model_id]
    
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
                "is_default": True if self.default_model_id == model_info.model_id else False,
                "model_creation": datetime.fromtimestamp(model_info.creation_timestamp / 1000).replace(microsecond=0),
                "model_flavors": list(model_info.flavors.keys()),
                "model_tags": model_info.tags,
            })
        
        return result
    
    def warm_model(self, model_id: str):
        if self._model_is_hot(model_id):
            self.logger.info(f"Model {model_id!r} is hot.")
            return
        
        model_path = self._get_model_path(model_id)
        model = SklearnModel(model_id, model_path)
        
        self.hot_models[model_id] = model
        self.cold_models.remove(model_id)
        
        self.logger.info(f"Model {model_id!r} warmed.")
    
    def cool_model(self, model_id: str):
        if self._model_is_cold(model_id):
            self.logger.info(f"Model {model_id!r} is cold.")
            return
        
        del self.hot_models[model_id]
        self.cold_models.add(model_id)
        
        self.logger.info(f"Model {model_id!r} cooled.")
    
    def set_default(self, model_id: str):
        if not self._model_is_hot(model_id):
            raise OperationalError(f"Model {model_id!r} is not hot and can't be the default.")
        
        self.default_model_id = model_id
        
        self.logger.info(f"Model {model_id!r} set as default.")
