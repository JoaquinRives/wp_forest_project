from api.application import create_app
from api.config.config import DevelopmentConfig, ProductionConfig


application = create_app(
    config_object=DevelopmentConfig)


if __name__ == '__main__':
    application.run('0.0.0.0', port=80)

