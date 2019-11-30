import os
import logging
import pathlib
import os
import sys

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Log file
LOG_FILE = PACKAGE_ROOT / 'log_file.log'

FEATURES = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3',
            '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40',
            '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51',
            '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62',
            '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73',
            '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84',
            '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94',
            'coordinate_x', 'coordinate_y']


class Config:
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = 'this-really-needs-to-be-changed'
    SERVER_PORT = 5000


class ProductionConfig(Config):
    DEBUG = False
    SERVER_ADDRESS: os.environ.get('SERVER_ADDRESS', '0.0.0.0')
    SERVER_PORT: os.environ.get('SERVER_PORT', '5000')


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
