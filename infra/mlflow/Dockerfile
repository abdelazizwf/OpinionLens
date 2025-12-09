FROM ghcr.io/mlflow/mlflow:v3.6.0

RUN apt update && \ 
    apt install -y gcc libpq-dev && \
    pip install psycopg2 scikit-learn && \
    mkdir -p /mlflow

RUN echo "Hello"

VOLUME [ "/mlflow" ]

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--backend-store-uri", \
    "postgresql://root:postgres@my-postgres.docker-net:5432/database", \
    "--serve-artifacts" \
    ]
