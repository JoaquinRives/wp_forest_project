import numpy as np
import pandas as pd
from api.config import config
import logging
from api.config import logger_config

_logger = logging.getLogger(__name__)
_logger = logger_config.set_logger(_logger)


def validate_inputs(input_data):
    """Check prediction inputs """

    input_data = input_data.copy()
    isValid = True
    errors = []
    try:
        # Convert json to DataFrame
        input_data = pd.DataFrame.from_dict(input_data)
        input_data.sort_index(axis=1, inplace=True)

        # Check for variables with NA
        if input_data.isnull().any().any():
            isValid = False
            errors.append("Input validation failed: Input Data contains missing values")
            _logger.error("Input validation failed: Input Data contains missing values")

        # Check for non-numerical values
        if not input_data.applymap(np.isreal).all().all():
            isValid = False
            errors.append("Input validation failed: Input Data contains non-numerical values")
            _logger.error("Input validation failed: Input Data contains non-numerical values")

        # Check names and number of features
        input_features = input_data.columns.values
        if (input_features != config.FEATURES).any():
            isValid = False
            errors.append(f"The name or numbers of features is incorrect:\n"
                          f"Model features: \n {config.FEATURES} \n"
                          f"Input features: \n {input_features} \n")
            _logger.error(f"The name or numbers of features is incorrect:\n"
                          f"Model features: \n {config.FEATURES} \n"
                          f"Input features: \n {input_features} \n")

    except Exception as e:
        isValid = False
        errors.append(f"Exception during validation: {e}")
        _logger.error(f"Exception during validation: {e}")

    _logger.debug(f"Input validation: \n"
                  f"    - input data: {input_data}\n"
                  f"    - isValid: {isValid}\n"
                  f"    - errors: {errors}")

    return input_data, isValid, errors
