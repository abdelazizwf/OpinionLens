from fastapi import FastAPI
from pydantic import BaseModel

from opinionlens.api.models import fetch_model_from_registery, make_prediction


class ModelFetchRequest(BaseModel):
    model_id: str
    model_name: str | None = None
    model_version: int | None = None
    set_current_model: bool = False


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


@app.get("/v1/predict")
async def predict(text: str):
    prediction = "POSITIVE" if make_prediction(text) == 1 else "NEGATIVE"
    return {"prediction": prediction}


@app.post("/v1/_/fetch_model")
async def fetch_model(request: ModelFetchRequest):
    model_path = fetch_model_from_registery(
        request.model_id, request.set_current_model
    )
    return {
        "message": f"Model saved at {model_path!r}",
    }
