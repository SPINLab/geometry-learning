"""
This script executes the task of estimating the building type, based solely on the geometry signature for that
building. The data for this script is committed in the repository and can be regenerated by running the
prep/get-data.sh and prep/preprocess-buildings.py scripts, which will take about an hour or two.

This script itself will run for about six hours depending on your hardware, if you have at least a recent i7 or
comparable.
"""

import os
import sys
import multiprocessing
from time import time
from datetime import datetime, timedelta

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from topoml_util.slack_send import notify

SCRIPT_VERSION = '0.0.1'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
NUM_CPUS = multiprocessing.cpu_count() - 1 or 1
DATA_FOLDER = SCRIPT_DIR + '/../../files/archaeology/'
FILENAME_PREFIX = 'archaeology_order_30_train'
SCRIPT_START = time()

if __name__ == '__main__':  # this is to squelch warnings on scikit-learn multithreaded grid search
    # Load training data
    training_files = []
    for file in os.listdir(DATA_FOLDER):
        if file.startswith(FILENAME_PREFIX) and file.endswith('.npz'):
            training_files.append(file)

    train_fourier_descriptors = np.array([])
    train_labels = np.array([])

    for index, file in enumerate(training_files):  # load and concatenate the training files
        train_loaded = np.load(DATA_FOLDER + file)

        if index == 0:
            train_fourier_descriptors = train_loaded['fourier_descriptors']
            train_labels = train_loaded['feature_type']
        else:
            train_fourier_descriptors = \
                np.append(train_fourier_descriptors, train_loaded['fourier_descriptors'], axis=0)
            train_labels = \
                np.append(train_labels, train_loaded['feature_type'], axis=0)

    scaler = StandardScaler().fit(train_fourier_descriptors)
    train_fourier_descriptors = scaler.transform(train_fourier_descriptors)

    C_range = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    degree_range = range(1, 7)
    param_grid = dict(degree=degree_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(
        SVC(kernel='poly', max_iter=int(1e7)),
        n_jobs=NUM_CPUS,
        param_grid=param_grid,
        verbose=10,
        cv=cv)

    print('Performing grid search on model...')
    print('Using %i threads for grid search' % NUM_CPUS)
    grid.fit(X=train_fourier_descriptors[::3], y=train_labels[::3])

    print("The best parameters are %s with a score of %0.3f"
          % (grid.best_params_, grid.best_score_))

    print('Training model on best parameters...')
    clf = SVC(kernel='poly',
              C=grid.best_params_['C'],
              degree=grid.best_params_['degree'],
              max_iter=int(1e7))
    clf.fit(X=train_fourier_descriptors, y=train_labels)

    # Run predictions on unseen test data to verify generalization
    print('Run on test data...')
    TEST_DATA_FILE = '../../files/archaeology/archaeology_order_30_test.npz'
    test_loaded = np.load(TEST_DATA_FILE)
    test_fourier_descriptors = test_loaded['fourier_descriptors']
    test_feature_type = np.asarray(test_loaded['feature_type'], dtype=int)
    test_fourier_descriptors = scaler.transform(test_fourier_descriptors)

    predictions = clf.predict(test_fourier_descriptors)
    accuracy = accuracy_score(test_feature_type, predictions)
    print('Test accuracy: %0.3f' % accuracy)

    runtime = time() - SCRIPT_START
    message = 'test accuracy of {} with C: {} degree: {} in {}'.format(
        str(accuracy), grid.best_params_['C'], grid.best_params_['degree'], timedelta(seconds=runtime))
    notify(SCRIPT_NAME, message)
    print(SCRIPT_NAME, 'finished successfully with', message)
