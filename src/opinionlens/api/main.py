from typing import Annotated

from fastapi import Body, FastAPI, HTTPException

from opinionlens.api.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.api.managers import model_manager
from opinionlens.api.routers import private

app = FastAPI()

app.include_router(private.router)


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
        model = model_manager.get_default_model()
        prediction = model.predict(text)
    except (ModelNotAvailableError, OperationalError) as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")
    
    prediction = "POSITIVE" if prediction == 1 else "NEGATIVE"
    return {"prediction": prediction}


@app.post("/v1/batch_predict")
async def batch_predict(
    batch: Annotated[list[str], Body()],
) -> list[str]:
    try:
        model = model_manager.get_default_model()
        predictions = model.batch_predict(batch)
    except (ModelNotAvailableError, OperationalError) as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")
    
    response = [
        "POSITIVE" if prediction == 1 else "NEGATIVE" for prediction in predictions
    ]
    
    return response
