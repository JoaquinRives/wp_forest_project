from sklearn.neighbors import KNeighborsRegressor
from feature_engine.outlier_removers import Winsorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from knn_model import preprocessors as pp
from knn_model.config import config

"""
Winsorizer: Preprocessor that removes outliers by capping the maximum of a distribution at a 
            fixed value.The maximum fixed value if determined using the inter-quantal range 
            proximity rule.
"""

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
