from contextlib import asynccontextmanager
from io import BytesIO

import pandas as pd
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware

from opinionlens.app.info import app_info
from opinionlens.app.middleware import log_error_responses
from opinionlens.app.routers import api
from opinionlens.app.routers.inference import batch_predict
from opinionlens.common.settings import get_settings

instrumentator = Instrumentator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global instrumentator
    instrumentator.expose(app)
    yield


app = FastAPI(
    **app_info, # type: ignore
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(BaseHTTPMiddleware, dispatch=log_error_responses)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Key"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts = [
        "abdelazizwf.dev", "*.abdelazizwf.dev",
        "localhost", "*.localhost",
        "docker-net", "*.docker-net",
    ]
)

app.include_router(
    api.router,
    prefix="/api/v1",
    tags=["api"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

instrumentator = instrumentator.instrument(app)

templates = Jinja2Templates(directory="static/html")


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


@app.post("/upload_txt")
async def upload_txt(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    delimiter: str = Form("\n"),
):
    """Predict the sentiments of texts in an uploaded file."""
    settings = get_settings()
    if file.size is not None and file.size > settings.api.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.api.max_file_size_mb}MB."
        )

    try:
        content = await file.read()
        text_content = content.decode("utf-8")

        # Handle the special case for escaped newline strings from forms
        if delimiter == "\\n":
            delimiter = "\n"
        elif delimiter == "\\t":
            delimiter = "\t"

        batch = [line.strip() for line in text_content.split(delimiter) if line.strip()]

        if not batch:
            raise HTTPException(status_code=400, detail="The uploaded file is empty or contains no valid text segments.")

        return await batch_predict(batch, background_tasks)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not decode file. Please upload a valid UTF-8 text file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_csv")
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    column_name: str = Form(...),
):
    """Predict sentiments of texts in a CSV file and append results."""
    settings = get_settings()
    if file.size is not None and file.size > settings.api.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.api.max_file_size_mb}MB."
        )

    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))

        if column_name not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column_name}' not found in the uploaded CSV."
            )

        # Remove empty rows in the specified column for prediction
        batch = df[column_name].astype(str).tolist()

        if not batch:
             raise HTTPException(status_code=400, detail="The specified column is empty.")

        predictions = await batch_predict(batch, background_tasks)
        df["sentiment"] = predictions

        # Save to buffer
        stream = BytesIO()
        df.to_csv(stream, index=False)
        stream.seek(0)

        return StreamingResponse(
            stream,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=evaluated_{file.filename}"}
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")
