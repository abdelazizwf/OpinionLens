from typing import Annotated

from fastapi import Body, FastAPI

from opinionlens.api.inference import fetch_model_from_registery, make_prediction

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


@app.post("/v1/batch_predict")
async def batch_predict(
    text_batch: Annotated[list[str], Body()],
) -> list[str]:
    response = []
    for text in text_batch:
        prediction = "POSITIVE" if make_prediction(text) == 1 else "NEGATIVE"
        response.append(prediction)
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
