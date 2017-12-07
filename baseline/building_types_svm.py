import os

from datetime import datetime

import numpy as np
from sklearn.svm import SVC

SCRIPT_VERSION = '0.0.2'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
DATA_FILE = '../files/neighborhoods.npz'

loaded = np.load(DATA_FILE)
fourier_descriptors = loaded['fourier_descriptors']
