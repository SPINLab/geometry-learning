import multiprocessing
import os
from datetime import datetime

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# This script executes the task of estimating the number of inhabitants of a neighborhood to be under or over the
# median of all neighborhoods, based solely on the geometry for that neighborhood. The data for this script can be
# generated by running the prep/get-data.sh and prep/preprocess-neighborhoods.py scripts, which will take about an
# hour or two.

# The script itself will run for about two hours depending on your hardware, if you have at least a recent i7 or
# comparable

SCRIPT_VERSION = '0.0.2'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
TRAINING_DATA_FILE = '../files/neighborhoods/neighborhoods_train.npz'
NUM_CPUS = multiprocessing.cpu_count() - 1 or 1
N_NEIGHBORS = 10

if __name__ == '__main__':  # this is to squelch warnings on scikit-learn multithreaded grid search
    train_loaded = np.load(TRAINING_DATA_FILE)
    train_fourier_descriptors = train_loaded['fourier_descriptors']
    train_above_or_below_median = train_loaded['above_or_below_median']

    scaler = StandardScaler().fit(train_fourier_descriptors)
    train_fourier_descriptors = scaler.transform(train_fourier_descriptors)
    clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)

    print('Fitting data to model...')
    scores = cross_val_score(clf, train_fourier_descriptors, train_above_or_below_median, cv=10, n_jobs=NUM_CPUS)
    print('Cross-validation scores:', scores)
    clf.fit(train_fourier_descriptors, train_above_or_below_median)

    # Run predictions on unseen test data to verify generalization
    TEST_DATA_FILE = '../files/neighborhoods/neighborhoods_test.npz'
    test_loaded = np.load(TEST_DATA_FILE)
    test_fourier_descriptors = test_loaded['fourier_descriptors']
    test_above_or_below_median = np.asarray(test_loaded['above_or_below_median'], dtype=int)
    test_fourier_descriptors = scaler.transform(test_fourier_descriptors)

    print('Run on test data...')
    predictions = clf.predict(test_fourier_descriptors)

    correct = 0
    for prediction, expected in zip(predictions, test_above_or_below_median):
        if all([pred == exp for pred, exp in zip(prediction, expected)]):
            correct += 1

    accuracy = correct / len(predictions)
    print('Test accuracy: %0.2f' % accuracy)