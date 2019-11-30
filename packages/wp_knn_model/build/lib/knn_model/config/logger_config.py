import logging
from knn_model.config import config

FORMAT = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —"
    "%(funcName)s:%(lineno)d — %(message)s")


def set_logger(logger):
    # Config level
    logging.basicConfig(
        level=logging.DEBUG)  # To log everything, by default it only logs warning and above.

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(config.LOG_FILE)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to the handlers
    c_handler.setFormatter(FORMAT)
    f_handler.setFormatter(FORMAT)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.propagate = False

    return logger
