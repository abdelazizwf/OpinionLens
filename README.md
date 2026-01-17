<p align="center" style="margin-bottom: 0px !important;">
  <img width="180" src="./static/images/logo.png" alt="OpinionLens logo" align="center">
</p>
<h1 align="center" style="margin-top: 0px;">OpinionLens</h1>

A production-ready sentiment analysis pipeline leveraging local ML training with tracked experiments, DVC data versioning and pipelines, remote MLflow model registry, Dockerized services, and Prometheus/Grafana monitoring.

## Description

This repository serves two distinct goals, the training and hyper-parameter tuning of sentiment analysis models, and running the application and API that serve those models.
The training is done locally, and makes use of several MLOps tools, such as [MLflow](https://mlflow.org/docs/latest/) for experiment tracking, [DVC](https://dvc.org/) for data versioning and running data pipelines to cache preprocessed data, and [Optuna](https://optuna.readthedocs.io/en/stable/index.html) for hyper-parameter tuning.
The application is a [FastAPI](https://fastapi.tiangolo.com/) server with a minimal frontend and API endpoints for inference and model management. The application also uses a remote MLflow model registry to manage and version models that were trained locally.

There are two modes of deployment. Local deployment, which runs the FastAPI server and the remote MLflow registry locally (in which case the "remote" MLflow registry is the same local MLflow server used for training).
The other is [Docker](https://www.docker.com/) deployment using `docker compose`, which runs the application and the remote MLflow registry as seperate containers. Docker deployment also containes a [PostgreSQL](https://www.postgresql.org/) container for any services requiring a database connection, a [Traefik](https://traefik.io/traefik) reverse proxy instance to regulate access to available services, and a fully featured monitoring stack with [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/) alongside various monitoring instrumentations, including for the FastAPI server and for inference operations. The goal of the Docker deployment is to make the application production-ready and easy to deploy on a VPS or a remote server (**This doesn't include security and authentication, which are extremely important for real-world deployment**).

## System Diagram

```mermaid
flowchart TD
    subgraph Local_Training[Local Training]
        A1[Training Scripts]
        A2[Local MLflow Tracking]
        A1 -->|Track Experiments| A2
    end

    subgraph MLflow[MLflow Remote Registry]
        B1[Dockerized MLflow Server]
    end

    A2 -->|Upload Model| B1

    subgraph Application[Application]
        C1[FastAPI Server]
        C2[Preprocessing Module]
        C2 -->|Preprocessing functions and objects| C1
        B1 -->|Load model| C1
    end

    subgraph Users[ ]
        D1[Users]
        D1 -->|Analysis Requests| C1
        C1 -->|Expose API & Frontend| D1
    end

    subgraph Monitoring[Monitoring Stack]
        E1[Prometheus]
        E2[Grafana Dashboards]
        E3[Node Exporter]
        E4[Postgres Exporter]
        C1 -->|Collect FastAPI & Inference Metrics| E1
        E3 -->|Collect Host Machine Metrics| E1
        E4 -->|Collect Database Metrics| E1
        E1 -->|Collect Metrics to be Displayed| E2
    end

    subgraph DB[Database]
        F1[Postgres Container]
        B1 --> F1
        F1 --> E4
    end

```

## Extra Information

### Data

- [IMDB Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews): Download and unpack at `/data/raw/IMDB Dataset/`.
- [Amazon Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews): Download and unpack at `/data/raw/Amazon Food Reviews/`.
- [Airline Tweets Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment): Download and unpack at `/data/raw/Airline Tweets/`.
