import pytest
from fastapi.testclient import TestClient

from opinionlens.api.main import app


@pytest.fixture(scope="module")
def test_app():
    client = TestClient(app)
    yield client


@pytest.fixture(scope="function")
def added_model_id(test_app):
    url = "/api/v1/_/models"
    body = {
        "model_uri": "basic_model/1",
        "set_default": True,
    }
    response = test_app.post(url, json=body)
    response_body = response.json()
    
    model_id = response_body["model_id"]
    yield model_id
    
    try:
        test_app.delete(f"{url}/{model_id}")
    finally:
        pass
