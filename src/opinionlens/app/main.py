from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
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
