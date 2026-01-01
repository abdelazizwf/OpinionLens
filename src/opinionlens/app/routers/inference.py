import time
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException

from opinionlens.app import instruments
from opinionlens.app.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.app.managers import model_manager

router = APIRouter()


@router.get("/predict")
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


@router.post("/predict")
async def encrypted_predict(
    text: Annotated[str, Body(embed=True)],
    background_tasks: BackgroundTasks
):
    """Predict the sentiment of a single text."""
    return await predict(text, background_tasks)


@router.post("/batch_predict")
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
