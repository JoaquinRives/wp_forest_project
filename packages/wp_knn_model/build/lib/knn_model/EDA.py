import numpy as np
import pandas as pd
import pandas_profiling
from sklearn.preprocessing import StandardScaler, Normalizer, QuantileTransformer, RobustScaler
from water_permeability_model.cross_validation import spatial_cross_validation, spatial_CV_knn_optimized
from timeit import default_timer as timer
from sklearn.model_selection import cross_val_score

# TODO
# read article
# TODO make a function that when you do a prediction it calculate the distance from the closest
#  neighhbors to the point you you are predicting and gives you and score of how reliable is that
#  prediction (the value of the score would be the score of the spatial-CV with a radius of the
#  same distance.

# borrar
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


# # ==> Standardization and DataFrame building
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
#
# df = pd.concat([input_data, coordinates], axis=1)
# df = pd.concat([df, output], axis=1)
#
#
# X = df.drop('target', axis=1)
# y = df['target']
#
# profile = pandas_profiling.ProfileReport(input_data.copy())
# profile.to_file("Titanic data profiling.html")

# ##############################################################
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
# start = timer()
#
# knn = KNeighborsRegressor(n_neighbors=3, metric='euclidean')
# result1 = spatial_CV_knn_optimized(model=knn, X=X, y=y, n_neighbors=3, radius=2500, scoring='c_index', verbose=True)
#
# end = timer()
# print(f"Time: {end - start}")
#
# start = timer()
#
#
# knn = KNeighborsRegressor(n_neighbors=3, metric='euclidean')
# result2 = spatial_cross_validation(model=knn, X=X, y=y, radius=2500, scoring='c_index', verbose=True)
#
# end = timer()
# print(f"Time: {end - start}")
#
#
# counter=0
# for p1, p2 in zip(result1, result2):
#     if p1 != p2:
#         print(counter)
#     counter+=1
# print(counter)


input_data = pd.read_csv('data/INPUT.csv', header=None)

# Standardization
scaler = StandardScaler()
input_data = pd.DataFrame(scaler.fit_transform(input_data))

coordinates = pd.read_csv('data/COORDINATES.csv', header=None)
coordinates.columns = ['coordinate_x', 'coordinate_y']

output = pd.read_csv('data/OUTPUT.csv', header=None)
output.columns = ['target']
output = output

df = pd.concat([input_data, coordinates], axis=1)
df = pd.concat([df, output], axis=1)


X = df.drop('target', axis=1)
y = df['target']
#

