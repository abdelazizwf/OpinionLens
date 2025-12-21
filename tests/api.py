import os

import pytest
import requests

BASE_URL = os.environ["TESTING_API_URL"]


def test_root_route():
    url = BASE_URL + "/"
    response = requests.get(url)
    
    assert response.status_code == 200
    assert type(response.json()) is dict


def test_about_route():
    url = BASE_URL + "/about"
    response = requests.get(url)
    
    assert response.status_code == 200
    assert type(response.json()) is dict

def test_add_model_route():
    url = BASE_URL + "/v1/_/models"
    body = {
        "model_uri": "basic_model/1",
        "set_default": True,
    }
    response = requests.post(url, json=body)
    
    assert response.status_code == 200
    assert type(response.json()) is dict


def test_list_models_route():
    url = BASE_URL + "/v1/_/models"
    response = requests.get(url)
    
    assert response.status_code == 200
    
    response_body = response.json()
    
    assert type(response_body) is list
    
    if response_body:
        element = response_body[0]
        
        assert type(element) is dict
        assert len(element) == 8


def test_single_prediction_route():
    url = BASE_URL + "/v1/predict"
    params = {"text": "I love this so much!"}
    response = requests.get(url, params=params)
    
    assert response.status_code == 200
    assert type(response.json()) is dict
    assert len(response.json()) == 1


def test_batch_prediction_route():
    url = BASE_URL + "/v1/batch_predict"
    body = [
        "I love this!", "This product is awful", "I really hate this man",
    ]
    response = requests.post(url, json=body)
    
    assert response.status_code == 200
    assert type(response.json()) is list
    assert len(response.json()) == len(body)


def test_delete_model_route():
    url = BASE_URL + "/v1/_/models"
    body = {
        "model_id": "m-beacc3a4a1624e92ba24bb4ffb349be1",
    }
    response = requests.delete(url, json=body)
    
    assert response.status_code == 200
    assert type(response.json()) is dict
