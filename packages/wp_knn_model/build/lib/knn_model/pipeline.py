from sklearn.neighbors import KNeighborsRegressor
from feature_engine.outlier_removers import Winsorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from knn_model import preprocessors as pp
from knn_model.config import config


pipe = Pipeline(
    [
        ('outlier_capper', Winsorizer(
                                variables=config.CONTINUOUS_VARS,
                                fold=config.WISORIZER_FOLD,
                                distribution='skewed',
                                tail='right')),
        ('variance_filter', pp.QuasiConstantFilter(
                                threshold=config.VARIANCE_THRESHOLD)),
        ('normalization', StandardScaler()),
        ('knn_model', KNeighborsRegressor(
                                n_neighbors=config.N_NEIGHBORS,
                                metric=config.METRIC))
    ]
)

# Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
#        19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
#        38, 39, 40, 41, 42, 44, 46, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59,
#        60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
#        79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
#       dtype='object')
#
# Index([8, 11, 20, 26, 31, 32, 40, 41, 46, 49, 53, 59, 70, 76, 90, 95, 96], dtype='object')
#
# Index([5, 21, 24, 26, 32, 37, 40, 59, 60, 63, 68, 70, 74, 75, 77, 94, 95], dtype='object')
#
# Index([1, 2, 10, 15, 17, 18, 21, 24, 25, 27, 32, 35, 38, 39, 40, 42, 46, 47,
#        49, 50, 53, 55, 56, 59, 60, 62, 64, 67, 68, 72, 73, 77, 79, 80, 81, 86,
#        88, 89, 90, 91, 92, 93, 94, 95, 96],
#       dtype='object')








