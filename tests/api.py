from .conftest import added_model_id, test_app


def test_root_route(test_app):
    url = "/api/v1"
    response = test_app.get(url)

    assert response.status_code == 200
    assert type(response.json()) is dict


def test_about_route(test_app):
    url = "/api/v1/about"
    response = test_app.get(url)

    assert response.status_code == 200
    assert type(response.json()) is dict


def test_add_model_route(test_app):
    url = "/api/v1/_/models"
    body = {
        "model_uri": "basic_model/1",
        "set_default": True,
    }
    response = test_app.post(url, json=body)

    assert response.status_code == 201

    response_body = response.json()

    assert type(response_body) is dict


def test_add_wrong_model(test_app):
    url = "/api/v1/_/models"
    body = {
        "model_uri": "nonexistent-model",
    }
    response = test_app.post(url, json=body)

    assert response.status_code == 503


def test_list_models_route(test_app, added_model_id):
    url = "/api/v1/_/models"
    response = test_app.get(url)

    assert response.status_code == 200

    response_body = response.json()

    assert type(response_body) is dict

    if response_body:
        element = list(response_body.values())[0]

        assert type(element) is dict
        assert len(element) == 7


def test_list_single_model_route(test_app, added_model_id):
    url = f"/api/v1/_/models/{added_model_id}"
    response = test_app.get(url)

    assert response.status_code == 200

    response_body = response.json()

    assert type(response_body) is dict
    assert len(response_body) == 7
    assert response_body["is_default"] is True


def test_list_wrong_model(test_app):
    url = "/api/v1/_/models/nonexistent-model"
    response = test_app.get(url)

    assert response.status_code == 404


def test_prediction_route(test_app, added_model_id):
    url = "/api/v1/predict"
    params = {"text": "I love this so much!"}
    response = test_app.get(url, params=params)

    assert response.status_code == 200

    response_body = response.json()

    assert type(response_body) is dict
    assert len(response_body) == 1


def test_prediction_with_no_model(test_app):
    url = "/api/v1/predict"
    params = {"text": "I love this"}
    response = test_app.get(url, params=params)

    assert response.status_code == 503


def test_encrypted_prediction_route(test_app, added_model_id):
    url = "/api/v1/predict"
    body = {"text": "I love this so much!"}
    response = test_app.post(url, json=body)

    assert response.status_code == 200

    response_body = response.json()

    assert type(response_body) is dict
    assert len(response_body) == 1

def test_batch_prediction_route(test_app, added_model_id):
    url = "/api/v1/batch_predict"
    body = [
        "I love this!", "This product is awful", "I really hate this man",
    ]
    response = test_app.post(url, json=body)

    assert response.status_code == 200

    response_body = response.json()

    assert type(response_body) is list
    assert len(response_body) == len(body)


def test_delete_model_route(test_app, added_model_id):
    url = f"/api/v1/_/models/{added_model_id}"
    response  = test_app.delete(url)

    assert response.status_code == 200
    assert type(response.json()) is dict


def test_delete_wrong_model(test_app):
    url = "/api/v1/_/models/nonexistent-model"
    response = test_app.delete(url)

    assert response.status_code == 404
