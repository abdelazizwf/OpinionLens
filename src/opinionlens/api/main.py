from typing import Annotated

from fastapi import Body, FastAPI, HTTPException

from opinionlens.api.exceptions import ModelNotAvailableError
from opinionlens.api.inference import (
    fetch_model_from_registery,
    list_local_models,
    make_prediction,
)

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
    try:
        prediction = make_prediction(text)
    except ModelNotAvailableError:
        return HTTPException(status_code=503)
    
    prediction = "POSITIVE" if prediction == 1 else "NEGATIVE"
    return {"prediction": prediction}


@app.post("/v1/batch_predict")
async def batch_predict(
    text_batch: Annotated[list[str], Body()],
) -> list[str]:
    response = []
    for text in text_batch:
        try:
            prediction = make_prediction(text)
        except ModelNotAvailableError:
            return HTTPException(status_code=503)
        
        response.append("POSITIVE" if prediction == 1 else "NEGATIVE")
    
    return response


@app.post("/v1/_/fetch_model")
async def fetch_model(
    model_id: Annotated[str, Body()],
    set_current_model: Annotated[bool, Body()] = False,
):
    model_path = fetch_model_from_registery(
        model_id, set_current_model
    )
    return {
        "message": f"Model saved at {model_path!r}",
    }


@app.get("/v1/_/list_models")
async def list_models():
    try:
        models = list_local_models()
    except ModelNotAvailableError:
        return HTTPException(status_code=503)
    
    return {"models": models}
