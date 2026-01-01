from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware

from opinionlens.app.info import app_info
from opinionlens.app.middleware import log_error_responses
from opinionlens.app.routers import api

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

app.include_router(
    api.router,
    prefix="/api/v1",
    tags=["api"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

instrumentator = instrumentator.instrument(app)

templates = Jinja2Templates(directory="static/html")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
