import numpy as np
from knn_model.predict import make_prediction
from knn_model.data_management import load_data
from knn_model.c_index import c_index
from sklearn.metrics import r2_score
from knn_model.config import logger_config
import logging

_logger = logging.getLogger(__name__)
_logger = logger_config.set_logger(_logger)


def test_make_single_prediction():
    # Given
    X_test, y_test = load_data('test')
    X_test_single = X_test[0:1]

    # When
    single_prediction = make_prediction(input_data=X_test_single)

    _logger.info(f"Single test prediction: {single_prediction.get('predictions')[0]}")

    # Then
    assert single_prediction is not None
    assert isinstance(single_prediction.get('predictions')[0], float)
    assert round(single_prediction.get('predictions')[0], 3) == 6.065


def test_make_multiple_predictions():
    # Given
    X_test, y_test = load_data('test')
    y_test = y_test['target'].values

    # When
    results = make_prediction(input_data=X_test)
    y_pred = results.get('predictions')
    c_index_score = c_index(list(y_test), list(y_pred))

    _logger.info(f"Multiple prediction test c_index_score: {c_index_score}")

    # Then
    assert results is not None
    assert len(y_pred) == len(X_test)
    assert round(c_index_score, 3) == 0.718
