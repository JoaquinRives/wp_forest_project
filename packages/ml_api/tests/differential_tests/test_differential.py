import math
from knn_model.config import config as model_config
from knn_model.predict import make_prediction
from knn_model.data_management import load_data
import pandas as pd
import pytest
from api.config import config


@pytest.mark.differential
def test_model_prediction_differential(*, save_file: str = 'test_data_predictions.csv'):
    """
    This test compares the similarity between the results of the current model and
    the previous model/version's results.
    """

    # Given
    # Load the saved previous model predictions
    previous_model_df = pd.read_csv(
        f"{config.PACKAGE_ROOT.parent.parent / 'ml_api/tests'}/{save_file}")
    previous_model_predictions = previous_model_df['predictions'].values

    X_test, y_test = load_data('test')

    # When
    current_result = make_prediction(X_test)
    current_model_predictions = current_result.get('predictions')

    # Then
    # diff the current model vs. the old model
    assert len(previous_model_predictions) == len(
        current_model_predictions)

    # Perform the differential test
    for previous_value, current_value in zip(
            previous_model_predictions, current_model_predictions):

        # convert numpy float64 to Python float.
        previous_value = previous_value.item()
        current_value = current_value.item()

        # rel_tol is the relative tolerance – it is the maximum allowed
        # difference between a and b, relative to the larger absolute
        # value of a or b. For example, to set a tolerance of 5%, pass
        # rel_tol=0.05.
        assert math.isclose(previous_value,
                            current_value,
                            rel_tol=model_config.ACCEPTABLE_MODEL_DIFFERENCE)
