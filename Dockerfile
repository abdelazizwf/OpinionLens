FROM python:3.13.3-slim

ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv

RUN mkdir /app && mkdir /app/models
WORKDIR /app

COPY .python-version params.yaml pyproject.toml uv.lock README.md LICENSE ./
COPY .env.prod ./.env
COPY src ./src
COPY objects ./objects

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

EXPOSE 80

CMD ["uv", "run", "uvicorn", "src.opinionlens.api.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "6"]
