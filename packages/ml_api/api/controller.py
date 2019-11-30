from flask import request, Blueprint, jsonify
from knn_model.predict import make_prediction
from api.validation import validate_inputs
from knn_model import __version__ as _version  # Version of the water_permeability_model package
from api import __version__ as api_version  # Version of the API
import logging
from api.config import logger_config

_logger = logging.getLogger(__name__)
_logger = logger_config.set_logger(_logger)

app = Blueprint('app', __name__)


@app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})


@app.route('/v1/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        _logger.debug(f'Inputs: {json_data}')

        # Step 2: Validate the input using marshmallow schema
        input_data, isValid, errors = validate_inputs(input_data=json_data)
        if isValid:
            # Step 3: Model prediction
            result = make_prediction(input_data=input_data)

            _logger.debug(f'Outputs: {result}')

            # Step 4: Convert numpy ndarray to list
            predictions = result.get('predictions')
            version = result.get('version')

        else:
            predictions = None
            version = _version
            _logger.debug()

        # Step 5: Return the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})


