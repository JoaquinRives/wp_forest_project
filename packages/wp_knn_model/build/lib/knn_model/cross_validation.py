import numpy as np
import pandas as pd
from math import sqrt
from knn_model.c_index import c_index
import random
from statistics import mean
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score, \
    accuracy_score, precision_score, recall_score, f1_score


# TODO add roc_auc_score
# TODO convert indexes to ignore to a sets
# TODO rename x -> X


def get_scoring_func(scoring):
    """Returns a scoring function"""

    default_scoring = c_index

    if scoring == "r2":
        scoring_func = r2_score
    elif scoring == "mse":
        scoring_func = mean_squared_error
    elif scoring == "expvar":
        scoring_func = explained_variance_score
    elif scoring == "c_index":
        scoring_func = c_index
    elif scoring == "accuracy":
        scoring_func = accuracy_score
    elif scoring == "precision":
        scoring_func = precision_score
    elif scoring == "recall":
        scoring_func = recall_score
    elif scoring == "f1":
        scoring_func = f1_score
    elif scoring == "c_index":
        scoring_func = c_index
    else:
        scoring_func = c_index
        print(f"Scoring argument '{scoring}' not known, "
              f"default scoring function ({default_scoring.__name__}) returned")

    return scoring_func


# TODO add this function to the jupyter notebook
def get_geographical_distance(coordinates_1, coordinates_2):
    """ Calculates the geographic distance between 2 pair of coordinates """
    x1, y1 = coordinates_1
    x2, y2 = coordinates_2
    distance = sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    return distance


def spatial_CV_knn_optimized(*, model, X, y, radius=0, scoring, n_neighbors=5, verbose=True):
    """  # TODO explanation
    model: sklearn.neighbors model (either for regression or classification)
    """

    # TODO comment:
    # The Time Complexity of the algorithm when testing n different values for the radius and
    # k different values for n_neighbors was O(n*k), while the Time Complexity of the algorithm
    # when using the optimized function is O(n). This means that we can test the prediction
    # performance with as many different number of neighbours as we want without practically
    # increasing the execution time.

    assert isinstance(X, pd.DataFrame), f"X: DataFrame type expected (Given: {type(X)})"
    assert isinstance(y, pd.Series) or isinstance(X, list), f"y: Series/list type expected " \
                                                            f"(Given: {type(y)})"

    dead_zone_radius = radius  # Distance in meters
    X = X.copy()
    y = y.copy()

    # Convert to a list if it is not a list
    if not isinstance(n_neighbors, list):
        n_neighbors = [n_neighbors]
    else:
        n_neighbors = n_neighbors

    # Set scoring function
    if scoring:
        if isinstance(scoring, str):
            scoring = get_scoring_func(scoring)
        else:
            # You can pass your own scoring function
            scoring = scoring

    # Create DataFrame to store the predictions
    df_predictions = pd.DataFrame([[None for n in n_neighbors] for i in range(len(X))])
    df_predictions.columns = [n for n in n_neighbors]

    # Create DataFrame to store the scoring results and predictions for each value of n_neighbors
    df_results = pd.DataFrame([None, None] for n in n_neighbors)
    df_results.columns = ['score', 'predictions']
    df_results.index = [n for n in n_neighbors]
    df_results.index.name = 'n_neighbours'

    # The kneighbours() method of the sklearn.kneighbors models returns a matrix where
    # each row is the indexes in X of the n_neighbors sorted in order of proximity to the
    # same sample/row in X.

    numberOfNeighbors = len(X) - 1  # We want to return all of the neighbors

    # We must drop the geographic coordinates when fitting the model
    model.fit(X.drop(['coordinate_x', 'coordinate_y'], axis=1), y)
    neighbors_matrix = model.kneighbors(n_neighbors=numberOfNeighbors, return_distance=False)

    # Now we have a matrix with the indexes of all the neighbours of each sample in ordered by proximity.
    # However, this matrix contains also those data points that are inside the dead zone radius.
    # To apply spatial leave-one-out cross-validation we need to find out which are the indexes of those data points
    # that are inside the dead zone so we can ignore them:
    for index, sample in X.iterrows():

        if radius != 0:
            # Get coordinates of the current sample
            sample_xy = (sample['coordinate_x'], sample['coordinate_y'])

            # Calculate geographic distances from each data point to the current sample
            X['distance'] = X.apply(
                lambda x: get_geographical_distance(sample_xy, (x['coordinate_x'], x['coordinate_y'])),
                axis=1)
            # Get indexes of the data points that are inside the dead zone radius
            inside = X[X['distance'] <= dead_zone_radius]
            indexesToIgnore = list(inside.index.values)
        else:
            indexesToIgnore = []

        # Get the list of neighbors of the current sample.
        neighbors = neighbors_matrix[index]

        # Get the indexes of the k nearest neighbours ignoring those inside the dead zone radius
        neighbors_indexes = []
        pointer = 0
        while len(neighbors_indexes) < max(n_neighbors):
            neighbor_index = neighbors[pointer]
            if neighbor_index in indexesToIgnore:
                pointer += 1
            else:
                neighbors_indexes.append(neighbor_index)
                pointer += 1

        # Make a prediction for each of the n_neighbors values
        for n in n_neighbors:
            df_predictions.loc[index][n] = mean([y[neighbor_index] for neighbor_index in neighbors_indexes[:n]])

    # Score the predictions
    for n in n_neighbors:
        y_pred = df_predictions[n].values
        #y_pred = [round(x, 5) for x in y_pred]
        score = scoring(y, y_pred)
        df_results.loc[n]['score'] = score
        df_results.loc[n]['predictions'] = y_pred

        if verbose:
            print(f"Neighbors: {n} - Score: {score}")

    return df_results

# TODO add an argument for the name of the columns with the coordinates
def spatial_cross_validation(*, model, X, y, scoring, radius=0, verbose=True):
    """  # TODO explanation"""
    dead_zone_radius = radius  # Distance in meters
    X = X.copy()
    y = y.copy()

    # Set scoring function
    if scoring:
        if isinstance(scoring, str):
            scoring = get_scoring_func(scoring)
        else:
            # You can pass your own scoring function
            scoring = scoring

    y_df = pd.DataFrame(y, columns=['target'])
    df = pd.concat([X, y_df], axis=1)

    # # This is just to follow the running progress of the function (if verbose=True)
    # total = len(df)
    # progress = 0

    predictions = []
    for index, sample in df.iterrows():
        sample_xy = (sample['coordinate_x'], sample['coordinate_y'])

        # Calculate geographic distances
        if radius != 0:
            df['distance'] = df.apply(
                lambda x: get_geographical_distance(sample_xy, (x['coordinate_x'], x['coordinate_y'])),
                axis=1)
            # Remove from the training data set the samples that are inside dead_zone_distance
            df_cleaned = df[df['distance'] > dead_zone_radius]

            # TODO ojo a esto
            X_train = df_cleaned.drop(['coordinate_x', 'coordinate_y', 'distance', 'target'], axis=1)

        else:
            # If radius==0 drop only the sample present in the test set
            df_cleaned = df.drop(index)
            X_train = df_cleaned.drop(['target'], axis=1)

        y_train = df_cleaned['target']
        # X_test = sample.drop(['coordinate_x', 'coordinate_y', 'target'])
        # TODO ojo
        X_test = sample.drop(['coordinate_x', 'coordinate_y', 'target'])
        X_test = np.array(X_test).reshape(1, -1)

        # Make prediction
        pred = model.fit(X_train, y_train).predict(X_test)
        #pred = [round(x, 5) for x in pred]
        predictions.append(pred[0])

        # # Print progress
        # if verbose:
        #     progress += 1
        #     print(f"Progress: {progress}/{total} ")

    # Evaluate predictions
    score = scoring(y, predictions)  #TODO

    if verbose:
        print(f"\nScore: {score}\n")

    df_predictions = pd.DataFrame.from_records({'y_test': list(y), 'y_pred': predictions})
    dict_results = {'score': score, 'df_results': df_predictions}

    return dict_results


def cross_validation(*, model, X, y, cv=10, scoring='c_index', refit=False, verbose=True):
    """Performs a Cross-validation"""
    X = X.copy()
    y = y.copy()

    cv = cv

    # Set scoring function
    if scoring:
        if isinstance(scoring, str):
            scoring = get_scoring_func(scoring)
        else:
            # You can pass your own scoring function
            scoring = scoring

    k_folder = KFold(n_splits=cv, shuffle=True, random_state=0)
    results = []
    cv = 0
    for train_index, test_index in k_folder.split(X, y):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]

        y_pred = model.fit(X_train, y_train).predict(X_test)
        cv_score = scoring(y_test, y_pred)
        results.append(cv_score)

        cv += 1

    score = mean(results)

    if verbose:
        print(f"Score ({scoring.__name__}): {score}", "\n")

    results_dict = {
        "score": score,
        "results": results
    }

    return results_dict


def leave_n_out_cv(*, model, X, y, n_out=10, scoring='mse', verbose=True):
    """ Leave-n-out Cross-validation"""
    model = model
    X = X.copy()
    y = y.copy()

    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y, columns=['target'])
    else:
        y.columns = ['target']

    df = pd.concat([X, y], axis=1)

    scoring_func = get_scoring_func(scoring)

    indexes = list(X.index.values)
    random.Random(4).shuffle(indexes)

    n_out = n_out

    if len(X) % n_out == 0:
        number_of_samples = int(len(X) / n_out)
    else:
        number_of_samples = int((len(X) // n_out) + 1)

    predictions = {'y_test': [], 'y_pred': []}
    end_loop = False
    while not end_loop:

        for i in range(number_of_samples):

            # Check how many indexes are left
            if len(indexes) <= n_out:
                # Put the remaining items in the sample and leave the loop run only 1 time more
                sample = indexes
                end_loop = True  # This will stop the loop on the next round

            else:
                sample = [indexes.pop(0) for i in range(n_out)]

            train = df.drop(sample)
            X_train = train.drop(['target'], axis=1)
            y_train = train['target']
            X_test = df.loc[sample].drop(['target'], axis=1)
            y_test = df.loc[sample]['target']

            # Reshape X_test and y_test is they are a single sample
            if len(X) == 1:
                X_test = np.array(X_test).reshape(1, -1)

            y_pred = model.fit(X_train, y_train).predict(X_test)

            # TODO do this to the leave one out of lecture2 instead of making the average of each
            #  each round
            predictions['y_test'].append(list(y_test))
            predictions['y_pred'].append(list(y_pred))


    score = scoring_func(predictions['y_test'], predictions['y_pred'])

    if verbose:
        print(f"Score ({scoring_func.__name__}): {score}", "\n")

    results_dict = {
        "score": score,
        "predictions": predictions
    }
    return results_dict

# #######################################
# from sklearn.preprocessing import StandardScaler
#
# input_data = pd.read_csv('data/INPUT.csv', header=None)
#
# # Standardization
# scaler = StandardScaler()
# input_data = pd.DataFrame(scaler.fit_transform(input_data))
#
# coordinates = pd.read_csv('data/COORDINATES.csv', header=None)
# coordinates.columns = ['coordinate_x', 'coordinate_y']
#
# output = pd.read_csv('data/OUTPUT.csv', header=None)
# output.columns = ['target']
# output = output
#
# df = pd.concat([input_data, coordinates], axis=1)
# df = pd.concat([df, output], axis=1)
#
#
# X = df.drop('target', axis=1)
# y = df['target']
#
# from sklearn.neighbors import KNeighborsRegressor
#
# knn = KNeighborsRegressor(metric='euclidean', n_neighbors=37)
#
# # scores = cross_validation(model=knn, x=X, y=y, fold=10, scoring='mse')
#
# results = leave_n_out_cv(model=knn, X=X, y=y, n_out=4, scoring='c_index', verbose=True)
#
