import os

import sys
from sklearn.model_selection import ParameterGrid
from topoml_util.slack_send import notify

SCRIPT_VERSION = '0.0.9'
N_TIMES = 1

HYPERPARAMS = {
    # 'BATCH_SIZE': [512],
    # 'REPEAT_DEEP_ARCH': [1],
    # 'LSTM_SIZE': [64],
    # 'DENSE_SIZE': [32],
    # 'EPOCHS': [200],
    # 'LEARNING_RATE': [1e-4],
    # 'GEOM_SCALE': [1e0, 1e-1, 1e-2, 1e-3],
    # 'RECURRENT_DROPOUT': [0.10],
    # 'PATIENCE': [8, 16, 24, 32, 40],
    # 'EARLY_STOPPING': 1
}
grid = list(ParameterGrid(HYPERPARAMS))

scripts = [
    'archaeo_feature_type_decision_tree.py',
    'archaeo_feature_type_knn.py',
    'archaeo_feature_type_logistic_regression.py',
    'archaeo_feature_type_svm_rbf.py',
    'building_type_decision_tree.py',
    'building_type_knn.py',
    'building_type_logistic_regression.py',
    'building_type_svm_rbf.py',
    'neighborhood_inhabintants_decision_tree.py',
    'neighborhood_inhabintants_knn.py',
    'neighborhood_inhabintants_logistic_regression.py',
    'neighborhood_inhabintants_svm_rbf.py',
]

for configuration in grid:
    # Set environment variables (this allows you to do hyperparam searches from any scripting environment)
    for key, value in configuration.items():
        os.environ[key] = str(value)

    # repeat to get a sense of results spread
    for _ in range(N_TIMES):
        for script in scripts:
            print('Executing', script)
            r_code = os.system('python3 {}'.format(script))
            if not r_code == 0:
                print('{} exited with error'.format(script))
                notify('{} grid search'.format(script), 'with error')
                sys.exit(1)

notify('All grid search', 'no errors')
print('All grid search', 'finished successfully')
