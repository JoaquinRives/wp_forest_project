# from  .config import config
# from  import data_management
# from  import preprocessors
# from  import pipeline
# from  import train_pipeline
# from  import predict

import os
from knn_model.config import config

# __version__ = config._version

with open(os.path.join(config.PACKAGE_ROOT, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()