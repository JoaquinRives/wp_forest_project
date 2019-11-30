import pandas as pd
from knn_model.config import config, logger_config
from knn_model.data_management import load_pipeline
from knn_model.validation import validate_inputs
import typing as t
import logging
from knn_model import __version__ as _version

_logger = logging.getLogger(__name__)
_logger = logger_config.set_logger(_logger)

pipeline_file_name = f'{config.MODEL_NAME}{_version}.pkl'
_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(input_data: t.Union[pd.DataFrame, dict]) -> dict:
    """ Make a prediction using a saved model pipeline.
        Args:
            input_data: Array of model prediction inputs.
        Returns:
            Predictions for each input row, as well as the version of the model that
            was used to make the predictions.
    """
    input_data = input_data.copy()
    input_data = pd.DataFrame(input_data)

    validated_data = validate_inputs(input_data=input_data)

    prediction = _pipe.predict(validated_data)
    prediction = [i[0] for i in prediction]

    results = {'predictions': prediction, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results
