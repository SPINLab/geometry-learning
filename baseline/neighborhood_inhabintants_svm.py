import os

from datetime import datetime

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC

SCRIPT_VERSION = '0.0.2'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
DATA_FILE = '../files/neighborhoods/neighborhoods_train.npz'

loaded = np.load(DATA_FILE)
fourier_descriptors = loaded['fourier_descriptors']
shape = fourier_descriptors.shape
fourier_descriptors = np.reshape(
    fourier_descriptors,
    (shape[0], shape[1] * shape[2]))
above_or_below_median = loaded['above_or_below_median'][:, 0]
above_or_below_median = np.reshape(above_or_below_median, (above_or_below_median.shape[0]))

# Copy-paste from http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples
# -svm-plot-rbf-parameters-py

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)

print('Performing grid search on model...')
grid.fit(X=fourier_descriptors, y=above_or_below_median)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X=fourier_descriptors, y=above_or_below_median)
        classifiers.append((C, gamma, clf))
