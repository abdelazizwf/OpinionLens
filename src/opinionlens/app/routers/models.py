from typing import Annotated

from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from mlflow.exceptions import MlflowException

from opinionlens.app.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.app.managers import model_manager, object_manager
from opinionlens.common.settings import get_settings

settings = get_settings()

router = APIRouter()


@router.post("/", status_code=201)
def fetch_model(
    model_uri: Annotated[str, Body()],
    set_default: Annotated[bool, Body()] = False,
):
    """Retrieve a new model from the model registry."""
    if not model_uri.startswith("models:/"):
        model_uri = "models:/" + model_uri

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
        except ModelNotAvailableError as e:
            raise HTTPException(status_code=404, detail=f"{type(e).__name__}: {e.message}")
        except OperationalError as e:
            raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")

        message += " and set as default"

    return {
        "model_id": model_id,
        "message": message,
    }


@router.get("/")
async def list_models():
    """List the details of all available models."""
    models = model_manager.get_model_info()
    return models


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


@router.post("/objects")
def upload_model_related_object(file: UploadFile = File(...)):
    try:
        object_manager.add_object(file)

    except OperationalError:
            raise HTTPException(
                status_code=400,
                detail=f"Object {file.filename} already exists. Consider deleting it first."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during upload: {e}")
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename!r}"}


@router.delete("/objects")
async def delete_model_related_object(filename: str):
    try:
        object_manager.delete_object(filename)
    except OperationalError as e:
        raise HTTPException(status_code=404, detail=e.message)

    return {"message": f"Successfully deleted object {filename!r}"}


@router.get("/objects")
async def list_objects():
    return object_manager.get_objects()
