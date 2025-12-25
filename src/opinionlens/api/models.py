import os

import mlflow
import numpy as np
from scipy.sparse import spmatrix

from opinionlens.common.utils import get_logger
from opinionlens.preprocessing import clean_text, get_saved_tfidf_vectorizer, tokenizer

SAVED_MODEL_PATH = os.environ["API_SAVED_MODEL_PATH"]
LOGGING_LEVEL = os.environ["LOGGING_LEVEL"]


class Model:
    """Abstract class for model objects."""
    
    def predict(self, text: str):
        raise NotImplementedError()
    
    def batch_predict(self, batch: list[str]):
        raise NotImplementedError()


class SklearnModel(Model):
    """A class for Scikit-learn models.
    
    Attributes:
        model_id (str): The ID of the model in the registry.
        pyfunc_model (mlflow.pyfunc.PyFuncModel): The mlflow model object with functional interface.
    """
    
    def __init__(self, model_id: str, model_path: str):
        """
        Args:
            model_id: The ID of the model.
            model_path: The path of the model directory.
        """
        self.model_id = model_id
        self.pyfunc_model = mlflow.pyfunc.load_model(model_path)
        self._logger = get_logger(self.__class__.__name__, level=LOGGING_LEVEL)
    
    def preprocess_text(self, text: str) -> spmatrix:
        """Preprocess the input text.
        
        Args:
            text: The input text.
        
        Returns:
            The text encoding to be used as input to the model.
        """
        tokenized_text = " ".join(tokenizer(clean_text(text)))

        vectorizer = get_saved_tfidf_vectorizer()
        vectors = vectorizer.transform(np.array([tokenized_text]))
        
        self._logger.debug("Preprocessing done.")
        return vectors
    
    def predict(self, text: str) -> int:
        """Predict the sentiment of the input text.
        
        Args:
            text: The input text.
        
        Returns:
            Either 0 for negative sentiment, or 1 for positive sentiment.
        """
        self._logger.debug(f"Asked to predict {text!r}.")
        vectors = self.preprocess_text(text)
        prediction = int(self.pyfunc_model.predict(vectors)[0])
        self._logger.debug(f"Prediction result is {prediction!r}.")
        return prediction
    
    def batch_predict(self, batch: list[str]) -> list[int]:
        """Predict the sentiments of the provided texts.
        
        Args:
            batch: A list of input texts.
        
        Returns:
            A list of predictions, with 0 for negative sentiment, and 1 for positive sentiment.
        """
        self._logger.debug(f"Asked to batch predict a list of length {len(batch)!r}.")
        predictions = [self.predict(text) for text in batch]
        return predictions
