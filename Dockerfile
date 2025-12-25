FROM python:3.13.3-slim

RUN pip install uv

RUN mkdir /app && mkdir /app/models
WORKDIR /app

COPY .python-version params.yaml pyproject.toml uv.lock README.md LICENSE ./
COPY .env.prod ./.env
COPY src ./src
COPY objects ./objects

RUN uv sync --no-dev

EXPOSE 80

CMD ["uv", "run", "uvicorn", "src.opinionlens.api.main:app", "--host", "0.0.0.0", "--port", "80"]
