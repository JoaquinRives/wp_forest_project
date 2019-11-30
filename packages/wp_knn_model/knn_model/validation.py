import numpy as np
import pandas as pd


def validate_inputs(input_data: pd.DataFrame):
    """ Check model inputs """

    validated_data = input_data.copy()

    # Check for variables with NA
    assert not input_data.isnull().any().any(), \
        "Input validation failed: Input Data contains missing values"

    # Check for non-numerical values
    assert input_data.applymap(np.isreal).all().all(), \
        "Input validation failed: Input Data contains non-numerical values"

    return validated_data
