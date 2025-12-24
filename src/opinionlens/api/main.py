from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from opinionlens.api.exceptions import ModelNotAvailableError, OperationalError
from opinionlens.api.info import app_info
from opinionlens.api.managers import model_manager
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

app.include_router(private.router)

instrumentator = instrumentator.instrument(app)


@app.get("/api/v1")
async def root():
    return {"message": "Welcome to OpinionLens!"}


@app.get("/api/v1/about")
async def about():
    return app_info


@app.get("/api/v1/predict")
async def predict(text: str):
    """Predict the sentiment of a single text."""
    try:
        model = model_manager.get_default_model()
        prediction = model.predict(text)
    except (ModelNotAvailableError, OperationalError) as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")
    
    prediction = "POSITIVE" if prediction == 1 else "NEGATIVE"
    return {"prediction": prediction}


@app.post("/api/v1/predict")
async def encrypted_predict(text: Annotated[str, Body(embed=True)]):
    """Predict the sentiment of a single text."""
    return await predict(text)


@app.post("/api/v1/batch_predict")
async def batch_predict(
    batch: Annotated[list[str], Body()],
) -> list[str]:
    """Predict the sentiments of multiple texts."""
    try:
        model = model_manager.get_default_model()
        predictions = model.batch_predict(batch)
    except (ModelNotAvailableError, OperationalError) as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e.message}")
    
    response = [
        "POSITIVE" if prediction == 1 else "NEGATIVE" for prediction in predictions
    ]
    
    return response
