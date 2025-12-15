from typing import Annotated

from fastapi import Body, FastAPI, HTTPException

from opinionlens.api.exceptions import ModelNotAvailableError
from opinionlens.api.inference import (
    fetch_model_by_id,
    fetch_model_by_name,
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


@app.post("/v1/_/fetch_model/id")
async def fetch_model_id(
    model_id: Annotated[str, Body()],
    set_current_model: Annotated[bool, Body()] = False,
):
    model_path = fetch_model_by_id(
        model_id, set_current_model
    )
    return {
        "message": f"Model saved at {model_path!r}",
    }


@app.post("/v1/_/fetch_model/name")
async def fetch_model_name(
    model_name: Annotated[str, Body()],
    model_alias: Annotated[str | None, Body()] = None,
    model_version: Annotated[int | None, Body()] = None,
    set_current_model: Annotated[bool, Body()] = False,
):
    if model_alias is None and model_version is None:
        return HTTPException(
            status_code=400, detail="Either model_alias or model_version has to be provided."
        )
    
    model_path = fetch_model_by_name(
        model_name, model_alias, model_version, set_current_model
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
