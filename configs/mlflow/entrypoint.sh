#!/bin/sh

mlflow server \
    --host 0.0.0.0 \
    --backend-store-uri "postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@postgres.docker-net:5432/mlflow" \
    --serve-artifacts \
    --allowed-hosts "*.docker-net:*,*.localhost,localhost:*,*.abdelazizwf.dev" \
    --cors-allowed-origins "http://*.docker-net,http://*.localhost,https://*.abdelazizwf.dev,http://mlflow.docker-net,http://mlflow.localhots,https://mlflow.abdelazizwf.dev"
