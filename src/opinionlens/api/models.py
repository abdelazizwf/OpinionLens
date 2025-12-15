import os
from datetime import datetime

import mlflow
import numpy as np

from opinionlens.api.exceptions import ModelNotAvailableError
from opinionlens.preprocessing import clean_text, get_saved_tfidf_vectorizer, tokenizer

SAVED_MODEL_PATH = os.environ["API_SAVED_MODEL_PATH"]


class SklearnModel:
    
    def __init__(self, model_id: str, model_path: str):
        self.model_id = model_id
        self.pyfunc_model = mlflow.pyfunc.load_model(model_path)
    
    def preprocess_text(self, text: str):
        tokenized_text = " ".join(tokenizer(clean_text(text)))

        vectorizer = get_saved_tfidf_vectorizer()
        vectors = vectorizer.transform(np.array([tokenized_text]))
        
        return vectors
    
    def predict(self, text: str) -> int:
        vectors = self.preprocess_text(text)
        prediction = int(self.pyfunc_model.predict(vectors)[0])
        return prediction
    
    def batch_predict(self, batch: list[str]) -> list[int]:
        predictions = [self.predict(text) for text in batch]
        return predictions


class ModelManager:
    
    def __init__(self):
        self.hot_models = {}
        self.hot_model_chance = {}
        self.cold_models = {info["model_id"] for info in self.list_models()}
        self.default_model_id = None
    
    def _get_model_path(self, model_id: str):
        return os.path.join(SAVED_MODEL_PATH, model_id)
    
    def _download_model(self, model_uri: str, model_id: str):
        dst_path = self._get_model_path(model_id)
        
        if not os.path.exists(dst_path):
            dst_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=dst_path,
            )
        
        self.cold_models.add(model_id)
        
        return dst_path
    
    def get_default_model(self) -> SklearnModel:
        assert self.default_model_id is not None
        assert self.default_model_id in self.hot_models.keys()
        return self.hot_models[self.default_model_id]
    
    def get_sampled_model(self):
        raise NotImplementedError()
    
    def fetch_model_by_id(self, model_id):
        model_uri = f"models:/{model_id}"
        dst_path = self._download_model(model_uri, model_id)
        return dst_path
    
    def fetch_model_by_name(self, model_name, model_version=None, model_alias=None):
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
    
    def list_models(self, hot_only=False):
        _, ids, _ = next(iter(os.walk(SAVED_MODEL_PATH)))

        result = []
        for model_id in ids:
            if hot_only and model_id in self.cold_models:
                continue
            
            model_info = mlflow.models.get_model_info(
                self._get_model_path(model_id)
            )
            result.append({
                "model_name": model_info.name,
                "model_id": model_info.model_id,
                "model_creation": datetime.fromtimestamp(model_info.creation_timestamp / 1000).replace(microsecond=0),
                "model_flavors": list(model_info.flavors.keys()),
                "model_tags": model_info.tags,
            })
        
        return result
    
    def warm_model(self, model_id: str):
        assert model_id in self.cold_models
        model_path = self._get_model_path(model_id)
        model = SklearnModel(model_id, model_path)
        self.hot_models[model_id] = model
        self.cold_models.remove(model_id)
    
    def cool_model(self, model_id: str):
        assert model_id in self.hot_models.keys()
        del self.hot_models[model_id]
        self.cold_models.add(model_id)
    
    def set_default(self, model_id):
        assert model_id in self.hot_models.keys()
        self.default_model_id = model_id
