import os
from functools import lru_cache

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    HttpUrl,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLflowSettings(BaseModel):
    """Override with environment variables prefixed by `MLFLOW__`. Ex: `local_key` -> `MLFLOW__LOCAL_KEY`."""
    local_experiment_name: str = Field(
        "imdb_amazon_airtweets_data",
        description="The local active MLflow experiment",
    )
    local_tracking_uri: HttpUrl = Field(
        "http://localhost:5000",
        description="The url of the local MLflow server",
    )
    remote_tracking_uri: HttpUrl = Field(
        "http://localhost:5000",
        description="The url of the remote MLflow registry",
    )
    remote_experiment_name: str = Field(
        "OpinionLens",
        description="The remote active MLflow experiment",
    )

    @field_validator('remote_tracking_uri', 'local_tracking_uri')
    @classmethod
    def strip_trailing_slash(cls, v: HttpUrl) -> str:
        return str(v).rstrip('/')


class APISettings(BaseModel):
    """Override with environment variables prefixed by `API__`. Ex: `local_key` -> `API__LOCAL_KEY`."""
    saved_model_path: DirectoryPath = Field(
        "./models",
        description="The path to the models fetched by the API from the model registry",
    )
    logging_level: str = Field(
        "DEBUG",
        description="The logging level for the API",
    )


class Settings(BaseSettings):
    mlflow: MLflowSettings = MLflowSettings()
    api: APISettings = APISettings()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache
def get_settings():
    match os.environ.get("ENV", "local").lower():
        case "local":
            return Settings()
        case "stage":
            return Settings(_env_file=".env.stage")
        case "prod":
            return Settings(_env_file=".env.prod")
        case _:
            return Settings()
