import mlflow
import numpy as np
from fastapi import FastAPI

from opinionlens.preprocessing import clean_text, get_saved_tfidf_vectorizer, tokenizer

model = mlflow.sklearn.load_model(
    "models:/sklearn-lin_svc/1"
)


def make_prediction(text: str):
    tokenized_text = " ".join(tokenizer(clean_text(text)))

    vectorizer = get_saved_tfidf_vectorizer()
    vectors = vectorizer.transform(np.array([tokenized_text]))

    prediction = int(model.predict(vectors)[0])
    return prediction


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to OpinionLens!"}


@app.get("/about")
async def about():
    return {
        "name": "OpinionLens",
        "version": "0.0.2",
        "author": "Abdelaziz W. Farahat",
        "description": "A production-ready sentiment analysis pipeline leveraging local ML training, DVC data versioning, MLflow model registry, Dockerized inference, and Prometheus/Grafana monitoring.",
    }


@app.get("/api/v1/predict")
async def predict(text: str):
    prediction = "POSITIVE" if make_prediction(text) == 1 else "NEGATIVE"
    return {"prediction": prediction}
