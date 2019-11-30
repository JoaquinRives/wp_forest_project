import pandas as pd
from knn_model.predict import make_prediction
from knn_model.data_management import load_data
from knn_model.config import config

"""
Script to capture the predictions of the current model/version for the differential test.

Every time a new commit is pushed to the repository the differential test will compare the
captured results of the previous model/version against the results of the new model.

In order to not brake this logic this script should be run automatically at the end of the 
CI (Continuous Integration) pipeline once all the other test have passed, so that the
prediction results of the most recent model will be available for comparison next time 
a new model/version is committed .
"""


def capture_predictions():
    """ Save the test data predictions to a file """

    save_file = 'test_data_predictions.csv'
    X_test, y_test = load_data('test')

    results = make_prediction(input_data=X_test)

    # Save predictions
    predictions = results['predictions']
    predictions_df = pd.DataFrame(predictions, columns=['predictions'])

    predictions_df.to_csv(f"{config.PACKAGE_ROOT.parent.parent / 'ml_api/tests'}/{save_file}")


if __name__ == '__main__':
    capture_predictions()
