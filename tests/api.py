import pytest  # noqa: F401
from fastapi.testclient import TestClient

from opinionlens.api.main import app

client = TestClient(app)

SAVED_MODEL_ID = None


def test_root_route():
    url = "/api/v1"
    response = client.get(url)
    
    assert response.status_code == 200
    assert type(response.json()) is dict


def test_about_route():
    url = "/api/v1/about"
    response = client.get(url)
    
    assert response.status_code == 200
    assert type(response.json()) is dict


def test_add_model_route():
    global SAVED_MODEL_ID
    
    url = "/api/v1/_/models"
    body = {
        "model_uri": "basic_model/1",
        "set_default": True,
    }
    response = client.post(url, json=body)
    
    assert response.status_code == 200
    
    response_body = response.json()
    
    assert type(response_body) is dict
    
    SAVED_MODEL_ID = response_body["model_id"]


def test_list_models_route():
    global SAVED_MODEL_ID
    
    url = "/api/v1/_/models"
    response = client.get(url)
    
    assert response.status_code == 200
    
    response_body = response.json()
    
    assert type(response_body) is list
    
    if response_body:
        element = response_body[0]
        
        assert type(element) is dict
        assert len(element) == 8


def test_list_single_model_route():
    global SAVED_MODEL_ID
    
    if not SAVED_MODEL_ID:
        return
    
    url = f"/api/v1/_/models/{SAVED_MODEL_ID}"
    response = client.get(url)
    
    assert response.status_code == 200
    
    response_body = response.json()
    
    assert type(response_body) is dict
    assert len(response_body) == 8
    assert response_body["model_id"] == SAVED_MODEL_ID
    assert response_body["is_default"] is True


def test_single_prediction_route():
    url = "/api/v1/predict"
    params = {"text": "I love this so much!"}
    response = client.get(url, params=params)
    
    assert response.status_code == 200
    
    response_body = response.json()
    
    assert type(response_body) is dict
    assert len(response_body) == 1


def test_encrypted_prediction_route():
    url = "/api/v1/predict"
    body = {"text": "I love this so much!"}
    response = client.post(url, json=body)
    
    assert response.status_code == 200
    
    response_body = response.json()
    
    assert type(response_body) is dict
    assert len(response_body) == 1

def test_batch_prediction_route():
    url = "/api/v1/batch_predict"
    body = [
        "I love this!", "This product is awful", "I really hate this man",
    ]
    response = client.post(url, json=body)
    
    assert response.status_code == 200
    
    response_body = response.json()
    
    assert type(response_body) is list
    assert len(response_body) == len(body)


def test_delete_model_route():
    global SAVED_MODEL_ID
    
    url = f"/api/v1/_/models/{SAVED_MODEL_ID}"
    response  = client.delete(url)
    
    assert response.status_code == 200
    assert type(response.json()) is dict
