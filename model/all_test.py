"""
Final test script for evaluation statistics
"""

import os

import sys
from sklearn.model_selection import ParameterGrid
from topoml_util.slack_send import notify

notify('ALL TEST SCRIPT RUNNING FINAL TESTS', 'STARTING')

SCRIPT_VERSION = '1.0.0'
N_TIMES = 1

HYPERPARAMS = {  # All using standard hyperparameters
    # 'BATCH_SIZE': [512],
    # 'REPEAT_DEEP_ARCH': [0],
    # 'LSTM_SIZE': [64],
    # 'DENSE_SIZE': [32],
    # 'EPOCHS': [200],
    # 'LEARNING_RATE': [1e-4],
    # 'GEOM_SCALE': [1e0, 1e-1, 1e-2, 1e-3],  # Leave at standard normalization
    # 'RECURRENT_DROPOUT': [0.10],
    # 'PATIENCE': [8, 16, 24, 32, 40],  # Early stopping disabled by default
}
grid = list(ParameterGrid(HYPERPARAMS))

scripts = [
    # 'neighborhood_convnet.py',
    # 'neighborhood_lstm.py',
    # 'building_convnet.py',
    # 'building_lstm.py',
    'archaeology_convnet.py',
    'archaeology_lstm.py'
]

for configuration in grid:
    # Set environment variables (this allows you to do hyperparam searches from any scripting environment)
    for key, value in configuration.items():
        os.environ[key] = str(value)

    # repeat to get a sense of results spread
    for _ in range(N_TIMES):
        for script in scripts:
            r_code = os.system('python3 {} --test'.format(script))
            if not r_code == 0:
                print('{} exited with error'.format(script))
                notify('{} grid search'.format(script), 'with error')
                sys.exit(1)

notify('ALL TEST', 'no errors')
print('ALL TEST', 'finished successfully')
