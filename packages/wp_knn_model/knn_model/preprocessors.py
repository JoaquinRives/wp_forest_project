import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class QuasiConstantFilter(BaseEstimator, TransformerMixin):
    """ Feature selection filter class for quasi-constant variables """

    def __init__(self, threshold):
        self.variance_threshold = threshold
        self.quasi_constant_feat = []

    def fit(self, X, y=None):
        X = X.copy()
        X = X.drop(['coordinate_x', 'coordinate_y'], axis=1)
        for variable in X.columns:
            # find the predominant value
            predominant = (X[variable].value_counts() / np.float(
                len(X))).sort_values(ascending=False).values[0]

            # evaluate predominant feature
            if predominant > self.variance_threshold:
                self.quasi_constant_feat.append(variable)

        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = X.drop(self.quasi_constant_feat, axis=1)
        return X
