import numpy as np
import pandas as pd
import joblib
from knn_model.config import config
from knn_model import __version__ as _version



def load_data(data):
    """ Load the training/test data """

    if data == 'all':
        input_data = pd.read_csv(config.INPUT_TRAIN, header=None)
        coordinates = pd.read_csv(config.COORDINATES_TRAIN, header=None)
        output = pd.read_csv(config.OUTPUT_TRAIN, header=None)

    elif data == 'train':
        input_data = pd.read_csv(config.INPUT_TRAIN, header=None)
        coordinates = pd.read_csv(config.COORDINATES_TRAIN, header=None)
        output = pd.read_csv(config.OUTPUT_TRAIN, header=None)
    elif data == 'test':
        input_data = pd.read_csv(config.INPUT_TEST, header=None)
        coordinates = pd.read_csv(config.COORDINATES_TEST, header=None)
        output = pd.read_csv(config.OUTPUT_TEST, header=None)
    else:
        raise Exception(f"Data ({data}) is not valid as an argument, the only valid arguments "
                        f"are 'all', 'train' and 'test' strings.")

    output.columns = ['target']

    coordinates.columns = ['coordinate_x', 'coordinate_y']
    X = pd.concat([input_data, coordinates], axis=1)

    # Convert column names to string (to avoid error when when loading data from a json)
    X.columns = [str(col_name) for col_name in X.columns.values]
    X.sort_index(axis=1, inplace=True)
    y = output

    return X, y


def save_pipeline(pipeline_to_persist):
    """ Persist the pipeline"""
    model_name = f'{config.MODEL_NAME}{_version}.pkl'
    save_path = config.TRAINED_MODEL_DIR / model_name
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name):
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model
