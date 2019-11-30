import pathlib

# Paths
PATH_CONFIG = pathlib.Path(__file__).resolve().parent
PACKAGE_ROOT = pathlib.Path(PATH_CONFIG).resolve().parent

DATA_DIR = PACKAGE_ROOT / 'data'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
MODEL_NAME = 'model_'

# Log file
LOG_FILE = PACKAGE_ROOT / 'log_file.log'

# Data files
INPUT = DATA_DIR / 'INPUT.csv'
COORDINATES = DATA_DIR / 'COORDINATES.csv'
OUTPUT = DATA_DIR / 'OUTPUT.csv'

# Training data files
INPUT_TRAIN = DATA_DIR / 'INPUT_TRAIN.csv'
COORDINATES_TRAIN = DATA_DIR / 'COORDINATES_TRAIN.csv'
OUTPUT_TRAIN = DATA_DIR / 'OUTPUT_TRAIN.csv'

# Testing data files
INPUT_TEST = DATA_DIR / 'INPUT_TEST.csv'
COORDINATES_TEST = DATA_DIR / 'COORDINATES_TEST.csv'
OUTPUT_TEST = DATA_DIR / 'OUTPUT_TEST.csv'

# Continuous numerical variables (we are considering continuous variables only those features
# with more than 9 different values.
CONTINUOUS_VARS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                   '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '28', '29',
                   '30', '31', '32', '33', '34', '35', '38', '39', '40', '41', '42', '43', '44',
                   '50', '51', '54', '60', '64', '65', '66', '67', '68', '69', '75', '85', '90',
                   '91', '92', '93', '94']

# Variance threshold of the QuasiConstantFilter preprocessor
VARIANCE_THRESHOLD = 0.01

# WInsorizer fold for outlier capping
WISORIZER_FOLD = 1.75

# k-Neighbors model
N_NEIGHBORS = 37
METRIC = 'euclidean'

# Differential test tolerance (acceptable difference in the predictions between old and new
# version of the model)
ACCEPTABLE_MODEL_DIFFERENCE = 0.05
