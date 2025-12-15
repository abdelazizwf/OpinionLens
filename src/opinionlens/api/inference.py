import os
from datetime import datetime

import mlflow
import numpy as np

from opinionlens.api.exceptions import ModelNotAvailableError
from opinionlens.preprocessing import clean_text, get_saved_tfidf_vectorizer, tokenizer

SAVED_MODEL_PATH = os.environ["API_SAVED_MODEL_PATH"]
CURRENT_MODEL_ID = None


def download_model(
    model_uri: str,
    model_id: str,
    set_current_model: bool = False
):
    global CURRENT_MODEL_ID
    
    dst_path = os.path.join(SAVED_MODEL_PATH, model_id)
    
    if not os.path.exists(dst_path):
        dst_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=dst_path,
        )
    else:
        print(f"Model {model_id!r} exists.")
    
    if set_current_model or not CURRENT_MODEL_ID:
        CURRENT_MODEL_ID = model_id
        get_model()
    
    return dst_path


def fetch_model_by_name(
    model_name: str,
    model_alias: str = None,
    model_version: int = None,
    set_current_model: bool = False
):
    assert model_alias or model_version, "Either model_alias or model_version must be specified."
    
    if model_alias:
        model_uri = f"models:/{model_name}@{model_alias}"
    else:
        model_uri = f"models:/{model_name}/{model_version}"
    
    model_info = mlflow.models.get_model_info(model_uri)
    dst_path = download_model(model_uri, model_info.model_id, set_current_model)
    return dst_path


def fetch_model_by_id(model_id: str, set_current_model: bool = False):
    global CURRENT_MODEL_ID
    
    model_uri = f"models:/{model_id}"
    dst_path = download_model(model_uri, model_id, set_current_model)
    
    return dst_path


def get_model():
    global CURRENT_MODEL_ID
    
    if CURRENT_MODEL_ID is None:
        raise ModelNotAvailableError(f"Model with id {CURRENT_MODEL_ID!r} doesn't exist.")
    
    dst_path = os.path.join(SAVED_MODEL_PATH, CURRENT_MODEL_ID)
    
    if not os.path.exists(dst_path):
        dst_path = fetch_model_by_id(CURRENT_MODEL_ID)
    
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
            "model_id": model_info.model_id,
            "model_creation": datetime.fromtimestamp(model_info.creation_timestamp / 1000).replace(microsecond=0),
            "model_flavors": list(model_info.flavors.keys()),
            "model_tags": model_info.tags,
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
