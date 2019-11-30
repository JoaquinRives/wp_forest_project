import api.config.config as config

with open(config.PACKAGE_ROOT / 'VERSION') as version_file:
    __version__ = version_file.read().strip()


