from typing import Annotated

from fastapi import Body, FastAPI, HTTPException

from opinionlens.api.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.api.models import ModelManager

model_manager = ModelManager()

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


@app.post("/v1/_/models/id")
async def fetch_model_id(
    model_id: Annotated[str, Body()],
    warm: Annotated[bool, Body()] = False,
    set_default: Annotated[bool, Body()] = False,
):
    model_path = model_manager.fetch_model_by_id(model_id)
    message = f"Model {model_id!r} saved at {model_path!r}"
    
    if warm or set_default:
        model_manager.warm_model(model_id)
        message += " and warmed"
    
    if set_default:
        try:
            model_manager.set_default(model_id)
        except (ModelNotAvailableError, OperationalError) as e:
            raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")
        
        message += " and set as default"
    
    return {
        "message": message,
    }


@app.post("/v1/_/models/name")
async def fetch_model_name(
    model_name: Annotated[str, Body()],
    model_alias: Annotated[str | None, Body()] = None,
    model_version: Annotated[int | None, Body()] = None,
    warm: Annotated[bool, Body()] = False,
    set_default: Annotated[bool, Body()] = False,
):
    if model_alias is None and model_version is None:
        raise HTTPException(
            status_code=400, detail="Either model_alias or model_version must be provided."
        )
    
    model_path, model_id = model_manager.fetch_model_by_name(model_name, model_version, model_alias)
    message = f"Model {model_id!r} saved at {model_path!r}"
    
    if warm or set_default:
        model_manager.warm_model(model_id)
        message += " and warmed"
    
    if set_default:
        try:
            model_manager.set_default(model_id)
        except (ModelNotAvailableError, OperationalError) as e:
            raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")
        
        message += " and set as default"
    
    return {
        "message": message,
    }


@app.get("/v1/_/models")
async def list_models(hot_only: bool = False):
    models = model_manager.list_models(hot_only)
    return {"models": models}
