from knn_model.config import config

with open(config.PACKAGE_ROOT / 'VERSION') as version_file:
    __version__ = version_file.read().strip()

from knn_model import data_management
from knn_model import preprocessors
from knn_model import pipeline
from knn_model import train_pipeline
from knn_model import predict



