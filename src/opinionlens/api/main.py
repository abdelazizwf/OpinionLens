import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware

from opinionlens.api import instruments
from opinionlens.api.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.api.info import app_info
from opinionlens.api.managers import model_manager
from opinionlens.api.middlewares import log_error_responses
from opinionlens.api.routers import private

instrumentator = Instrumentator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global instrumentator
    instrumentator.expose(app)
    yield


app = FastAPI(
    **app_info,
    lifespan=lifespan,
)

app.add_middleware(BaseHTTPMiddleware, dispatch=log_error_responses)

app.include_router(private.router)

instrumentator = instrumentator.instrument(app)


@app.get("/api/v1")
async def root():
    return {"message": "Welcome to OpinionLens!"}


@app.get("/api/v1/about")
async def about():
    return app_info


@app.get("/api/v1/predict")
async def predict(text: str, background_tasks: BackgroundTasks):
    """Predict the sentiment of a single text."""
    try:
        model = model_manager.get_default_model()

        start_time = time.perf_counter()
        prediction = model.predict(text)
        end_time = time.perf_counter()

    except (ModelNotAvailableError, OperationalError) as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")

    prediction = "POSITIVE" if prediction == 1 else "NEGATIVE"

    def log_metrics():
        instruments.INPUT_TEXT_LENGTH_CHARS.labels("/predict").observe(len(text))

        instruments.MODEL_INFERENCE_TIME_SECONDS.labels(
            "/predict",
            model.__class__.__name__,
        ).observe(end_time - start_time)

        instruments.PREDICTED_SENTIMENT_TOTAL.labels(
            prediction
        ).inc()

    background_tasks.add_task(log_metrics)

    return {"prediction": prediction}


@app.post("/api/v1/predict")
async def encrypted_predict(
    text: Annotated[str, Body(embed=True)],
    background_tasks: BackgroundTasks
):
    """Predict the sentiment of a single text."""
    return await predict(text, background_tasks)


@app.post("/api/v1/batch_predict")
async def batch_predict(
    batch: Annotated[list[str], Body()],
    background_tasks: BackgroundTasks,
) -> list[str]:
    """Predict the sentiments of multiple texts."""
    try:
        model = model_manager.get_default_model()

        start_time = time.perf_counter()
        predictions = model.batch_predict(batch)
        end_time = time.perf_counter()

    except (ModelNotAvailableError, OperationalError) as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")

    response = [
        "POSITIVE" if prediction == 1 else "NEGATIVE" for prediction in predictions
    ]

    def log_metrics():
        instruments.MODEL_INFERENCE_TIME_SECONDS.labels(
            "/batch_predict",
            model.__class__.__name__,
        ).observe(end_time - start_time)

        instruments.BATCH_INFERENCE_TIME_PER_ITEM_SECONDS.labels(
            "/batch_predict",
            model.__class__.__name__,
        ).observe((end_time - start_time) / len(batch))

        instruments.BATCH_SIZE_TEXT.labels(
            "/batch_predict"
        ).observe(len(batch))

        for text in batch:
            instruments.INPUT_TEXT_LENGTH_CHARS.labels("/batch_predict").observe(len(text))

        for prediction in response:
            instruments.PREDICTED_SENTIMENT_TOTAL.labels(
                prediction
            ).inc()

    background_tasks.add_task(log_metrics)

    return response


@app.get("/api/v1/metrics")
async def inference_metrics():
    return Response(
        generate_latest(registry=instruments.inference_registry),
        media_type=CONTENT_TYPE_LATEST,
    )
