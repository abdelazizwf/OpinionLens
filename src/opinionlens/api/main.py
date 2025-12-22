from typing import Annotated

from fastapi import Body, FastAPI, HTTPException

from opinionlens.api.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.api.managers import model_manager
from opinionlens.api.routers import private

description = """
A production-ready sentiment analysis pipeline leveraging local ML training, DVC data versioning, MLflow model registry, Dockerized services, and Prometheus/Grafana monitoring.

## Planned Features

- Monitoring stack with Prometheus and Grafana
- Reverse proxy configuration with Traefik
- Training and deploying deep models
- Model interpretability
- More raw data and using sampling techniques for training data
- Extensive testing
"""

app = FastAPI(
    title="OpinionLens",
    description=description,
    summary="AI-powered sentiment analysis with reproducible models and scalable deployment.",
    version="0.0.2",
    contact={
        "name": "Abdelaziz W. Farahat",
        "email": "abdelaziz.w.f@gmail.com",
    },
    license_info={
        "name": "Apache-2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

app.include_router(private.router)


@app.get("/api/v1")
async def root():
    return {"message": "Welcome to OpinionLens!"}


@app.get("/api/v1/about")
async def about():
    return {
        "name": "OpinionLens",
        "version": "0.0.2",
        "author": "Abdelaziz W. Farahat",
        "description": "A production-ready sentiment analysis pipeline leveraging local ML training, DVC data versioning, MLflow model registry, Dockerized inference, and Prometheus/Grafana monitoring.",
    }


@app.get("/api/v1/predict")
async def predict(text: str):
    try:
        model = model_manager.get_default_model()
        prediction = model.predict(text)
    except (ModelNotAvailableError, OperationalError) as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")
    
    prediction = "POSITIVE" if prediction == 1 else "NEGATIVE"
    return {"prediction": prediction}


@app.post("/api/v1/predict")
async def encrypted_predict(text: Annotated[str, Body(embed=True)]):
    return await predict(text)


@app.post("/api/v1/batch_predict")
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
