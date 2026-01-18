from datetime import datetime
from typing import Annotated

import mlflow
from fastapi import APIRouter, Body, HTTPException
from mlflow.exceptions import MlflowException

from opinionlens.app.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.app.managers import model_manager
from opinionlens.common.settings import get_settings

settings = get_settings()

router = APIRouter()


@router.post("/", status_code=201)
def fetch_model(
    model_name: Annotated[str, Body()],
    model_version: Annotated[int, Body()],
    set_default: Annotated[bool, Body()] = False,
):
    """Retrieve a new model from the model registry."""
    model_uri = f"models:/{model_name}/{model_version}"

    try:
        model_path, model_id = model_manager.fetch_model(model_uri)
    except MlflowException as e:
        raise HTTPException(status_code=503, detail=f"Model registry error:\n{e.message}")
    except OperationalError as e:
            raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")

    message = f"Model {model_id!r} saved at {model_path!r}"

    if set_default:
        try:
            model_manager.set_default(model_id)
        except OperationalError as e:
            raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")

        message += " and set as default"

    return {
        "model_id": model_id,
        "message": message,
    }


@router.get("/")
async def list_models(brief: bool = False):
    """List the details of all available models."""
    models = model_manager.get_model_info()

    if not brief:
        return models

    results = []
    for model_id, value in models.items():
        name_and_version = value["model_tags"]["mlflow.modelVersions"][0]
        results.append({
            "name": name_and_version["name"],
            "version": name_and_version["version"],
            "creation": value["model_creation"].isoformat(sep=" ", timespec="seconds"),
            "is_default": value["is_default"],
            "model_id": model_id,
        })

    return results


@router.get("/")
async def list_model(model_id: str):
    """List the details of a given model."""
    try:
        model = model_manager.get_model_info(model_id)
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=404, detail=f"{e.message}")

    return model


@router.delete("/")
async def delete_model(model_id: str):
    """Remove the given model from the backend."""
    try:
        model_manager.delete_model(model_id)
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=404, detail=f"{type(e).__name__}: {e.message}")
    except OperationalError as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")

    return {"message": f"Model {model_id!r} was deleted successfully."}


@router.get("/registry")
async def display_models_from_registry():
    """Display models available at the MLflow model registry."""
    models = mlflow.search_registered_models()

    results = []
    for model in models:
        if model.latest_versions[0].tags.get("experiment") == settings.mlflow.remote_experiment_name:
            results.append({
                "name": model.name,
                "latest_version": int(model.latest_versions[0].version),
                "latest_version_creation": datetime.fromtimestamp(
                    model.last_updated_timestamp / 1000
                ).isoformat(sep=" ", timespec="seconds"),
            })

    return results
