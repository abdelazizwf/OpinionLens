from fastapi import APIRouter, Depends # noqa
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from opinionlens.app import instruments
from opinionlens.app.dependencies import authenticate_admin # noqa
from opinionlens.app.info import app_info
from opinionlens.app.routers import inference, models

router = APIRouter()

router.include_router(
    inference.router,
    prefix="/inference",
    tags=["inference"],
)

router.include_router(
    models.router,
    prefix="/models",
    tags=["models"],
    # dependencies=[Depends(authenticate_admin)],
)


@router.get("/")
async def api_root():
    return {"message": "Welcome to OpinionLens!"}


@router.get("/about")
async def about():
    return app_info


@router.get("/metrics")
async def inference_metrics():
    return Response(
        generate_latest(registry=instruments.inference_registry),
        media_type=CONTENT_TYPE_LATEST,
    )
