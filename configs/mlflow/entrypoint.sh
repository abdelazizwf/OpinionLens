#!/bin/sh

mlflow server \
    --host 0.0.0.0 \
    --backend-store-uri "postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@postgres.docker-net:5432/database" \
    --serve-artifacts \
    --allowed-hosts "*.docker-net:*,*.localhost,localhost:*" \
    --cors-allowed-origins "http://*.docker-net,http://*.localhost"
