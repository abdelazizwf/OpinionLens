import os

import mlflow
import numpy as np
from scipy.sparse import spmatrix

from opinionlens.common.utils import get_logger
from opinionlens.preprocessing import clean_text, get_saved_tfidf_vectorizer, tokenizer

SAVED_MODEL_PATH = os.environ["API_SAVED_MODEL_PATH"]


class Model:
    
    def predict(self, text: str):
        raise NotImplementedError()
    
    def batch_predict(self, batch: list[str]):
        raise NotImplementedError()


class SklearnModel(Model):
    
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
