from typing import Annotated

from fastapi import APIRouter, Body, HTTPException
from mlflow.exceptions import MlflowException

from opinionlens.api.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.api.managers import model_manager

router = APIRouter(
    prefix="/api/v1/_",
    tags=["private"],
)


@router.post("/models")
async def fetch_model(
    model_uri: Annotated[str, Body()],
    set_default: Annotated[bool, Body()] = False,
):
    if not model_uri.startswith("models:/"):
        model_uri = "models:/" + model_uri
    
    try:
        model_path, model_id = model_manager.fetch_model(model_uri)
    except MlflowException as e:
        raise HTTPException(status_code=503, detail=f"Model registry error:\n{e.message}")
    
    message = f"Model {model_id!r} saved at {model_path!r}"
    
    if set_default:
        try:
            model_manager.set_default(model_id)
        except (ModelNotAvailableError, OperationalError) as e:
            raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")
        
        message += " and set as default"
    
    return {
        "model_id": model_id,
        "message": message,
    }


@router.get("/models")
async def list_models():
    models = model_manager.list_models()
    return models


@router.get("/models/{model_id}")
async def list_model(model_id: str):
    model = [m for m in model_manager.list_models() if m["model_id"] == model_id][0]
    return model


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
):
    try:
        model_manager.delete_model(model_id)
    except (ModelNotAvailableError, OperationalError) as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")
    
    return {"message": f"Model {model_id!r} was deleted successfully."}
