FROM ghcr.io/mlflow/mlflow:v3.6.0

RUN --mount=type=cache,target=/root/.cache/apt \
    apt update && \
    apt install -y gcc libpq-dev

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install psycopg2 scikit-learn mlflow[auth]

RUN mkdir -p /mlflow

VOLUME [ "/mlflow" ]

WORKDIR /mlflow

COPY entrypoint.sh /opt/entrypoint.sh

RUN chmod +x /opt/entrypoint.sh

EXPOSE 5000

ENTRYPOINT [ "/opt/entrypoint.sh" ]
