import os

import mlflow
import numpy as np

from opinionlens.preprocessing import clean_text, get_saved_tfidf_vectorizer, tokenizer

SAVED_MODEL_PATH = os.environ["API_SAVED_MODEL_PATH"]
CURRENT_MODEL_ID = None


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
    
    dst_path = os.path.join(SAVED_MODEL_PATH, CURRENT_MODEL_ID)
    
    if not os.path.exists(dst_path):
        fetch_model_from_registery(CURRENT_MODEL_ID)
    
    model = mlflow.pyfunc.load_model(dst_path)
    
    return model


def make_prediction(text: str):
    tokenized_text = " ".join(tokenizer(clean_text(text)))

    vectorizer = get_saved_tfidf_vectorizer()
    vectors = vectorizer.transform(np.array([tokenized_text]))
    
    model = get_model()

    prediction = int(model.predict(vectors)[0])
    return prediction


if __name__ == "__main__":
    get_model()
