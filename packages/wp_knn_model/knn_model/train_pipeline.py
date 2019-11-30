from knn_model import pipeline
from knn_model.data_management import load_data, save_pipeline
from knn_model.config import logger_config
import logging
from knn_model import __version__ as _version

_logger = logging.getLogger(__name__)
_logger = logger_config.set_logger(_logger)


def run_training() -> None:
    """ Train the model """
    # Load the training data
    X_train, y_train = load_data('train')

    # Train the model
    pipeline.pipe.fit(X_train, y_train)

    # Save the model
    save_pipeline(pipeline_to_persist=pipeline.pipe)

    _logger.info(f'Saving model version: {_version}')


if __name__ == '__main__':
    run_training()
