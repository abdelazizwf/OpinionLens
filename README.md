# OpinionLens

A production-ready sentiment analysis pipeline leveraging local ML training, DVC data versioning, MLflow model registry, Dockerized services, and Prometheus/Grafana monitoring.

## System Diagram

```mermaid
flowchart TD
    subgraph Local_Training[Local Training]
        A1[Training Scripts] --> A2[Local MLflow Tracking]
        A2 -->|Select Best Model| A3[Export Model Artifact]
    end

    subgraph MLflow[MLflow Registry]
        B1[Dockerized MLflow Server]
    end

    A3 -->|Upload Artifact| B1

    subgraph Inference[Inference API & Web Service]
        C1[FastAPI Container]
        C2[Preprocessing Module]
        C3[Load Model from MLflow]
        C1 --> C2
        C2 --> C3
        C3 --> C1
        C1 -->|Expose /predict & /metrics| D1[Clients]
        B1 --> C3
    end

    D1 -->|Requests| C1

    subgraph Monitoring[Monitoring Stack]
        E1[Prometheus]
        E2[Grafana Dashboards]
        E3[Node Exporter]
        E4[Postgres Exporter]
        C1 --> E1
        E3 --> E1
        E4 --> E1
        E1 --> E2
    end

    subgraph DB[Database]
        F1[Postgres Container]
        C1 --> F1
        B1 --> F1
        F1 --> E4
    end

```

## Extra Information

### Data

- [IMDB Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews): Download and unpack at `/data/raw/IMDB Dataset/`.
- [Amazon Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews): Download and unpack at `/data/raw/Amazon Food Reviews/`.
- [Airline Tweets Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment): Download and unpack at `/data/raw/Airline Tweets/`.
