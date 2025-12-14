import os
from datetime import datetime

import mlflow
import numpy as np

from opinionlens.api.exceptions import ModelNotAvailableError
from opinionlens.preprocessing import clean_text, get_saved_tfidf_vectorizer, tokenizer

SAVED_MODEL_PATH = os.environ["API_SAVED_MODEL_PATH"]
CURRENT_MODEL_ID = "m-bf9c6c1a0c2c42feba65d43fd15f0896"

def fetch_model_from_registery(model_id: str, set_current_model=False):
    global CURRENT_MODEL_ID
    
    model_uri = f"models:/{model_id}"
    dst_path = os.path.join(SAVED_MODEL_PATH, model_id)
    
    if not os.path.exists(dst_path):
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=dst_path,
        )
    
    if set_current_model or not CURRENT_MODEL_ID:
        CURRENT_MODEL_ID = model_id
    
    return dst_path


def get_model():
    global CURRENT_MODEL_ID
    
    if CURRENT_MODEL_ID is None:
        raise ModelNotAvailableError(f"Model with id {CURRENT_MODEL_ID!r} doesn't exist.")
    
    dst_path = os.path.join(SAVED_MODEL_PATH, CURRENT_MODEL_ID)
    
    if not os.path.exists(dst_path):
        dst_path = fetch_model_from_registery(CURRENT_MODEL_ID)
    
    model = mlflow.pyfunc.load_model(dst_path)
    
    return model


def list_local_models():
    _, ids, _ = next(iter(os.walk(SAVED_MODEL_PATH)))
    
    if ids == [] or ids is None:
        raise ModelNotAvailableError(f"No models were found in {SAVED_MODEL_PATH!r}.")
    
    result = []
    for id in ids:
        model_info = mlflow.models.get_model_info(
            os.path.join(SAVED_MODEL_PATH, id)
        )
        result.append({
            "model_name": model_info.name,
            "model_version": model_info.registered_model_version,
            "model_id": model_info.model_id,
            "model_creation": datetime.fromtimestamp(model_info.creation_timestamp / 1000).replace(microsecond=0),
        })
    return result


def make_prediction(text: str):
    tokenized_text = " ".join(tokenizer(clean_text(text)))

    vectorizer = get_saved_tfidf_vectorizer()
    vectors = vectorizer.transform(np.array([tokenized_text]))
    
    model = get_model()

    prediction = int(model.predict(vectors)[0])
    return prediction


if __name__ == "__main__":
    get_model()
