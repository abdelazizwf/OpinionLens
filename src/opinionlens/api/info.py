description = """
A production-ready sentiment analysis pipeline leveraging local ML training, DVC data versioning, MLflow model registry, Dockerized services, and Prometheus/Grafana monitoring.

## Planned Features

- Monitoring stack with Prometheus and Grafana
- Reverse proxy configuration with Traefik
- Training and deploying deep models
- Model interpretability
- More raw data and using sampling techniques for training data
- Extensive testing
"""

app_info = dict(
    title="OpinionLens",
    description=description,
    summary="AI-powered sentiment analysis with reproducible models and scalable deployment.",
    version="0.0.2",
    contact={
        "name": "Abdelaziz W. Farahat",
        "email": "abdelaziz.w.f@gmail.com",
    },
    license_info={
        "name": "Apache-2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)
