import json
from knn_model import __version__ as _version
from knn_model.data_management import load_data

from api import __version__ as api_version


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):
    # When
    response = flask_test_client.get('/version')

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == _version
    assert response_json['api_version'] == api_version


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # Given
    X_test, y_test = load_data('test')
    post_json = X_test[0:1].to_json(orient='records')

    # When
    response = flask_test_client.post('/v1/predict',
                                      json=json.loads(post_json))

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert round(prediction[0], 3) == 6.065

    assert response_version == _version
