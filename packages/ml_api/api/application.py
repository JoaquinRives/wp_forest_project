from flask import Flask
# from flask_basicauth import BasicAuth
from api.config.config import DevelopmentConfig, ProductionConfig
import logging
from api.config import logger_config

_logger = logging.getLogger(__name__)
_logger = logger_config.set_logger(_logger)


def create_app(*, config_object):
    """Create a flask app instance."""

    flask_app = Flask('app')
    flask_app.config.from_object(config_object)

    # # Basic Authentication ("Sign in" security to have access to the app)
    # basic_auth = BasicAuth(flask_app)

    # Import blueprints
    from api.controller import app
    flask_app.register_blueprint(app)
    _logger.info('Application instance created')

    return flask_app


if __name__ == '__main__':
    application = create_app(config_object=DevelopmentConfig)
    application.run()
