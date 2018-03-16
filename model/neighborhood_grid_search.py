import os
import socket

import sys

import numpy as np
from sklearn.model_selection import ParameterGrid
from topoml_util.slack_send import notify

SCRIPT_VERSION = '0.1.16'
N_TIMES = 6

HYPERPARAMS = {
    # 'BATCH_SIZE': [512],
    # 'REPEAT_DEEP_ARCH': [1, 2],
    # 'LSTM_SIZE': np.linspace(64, 128, 3, dtype=int),
    # 'DENSE_SIZE': [64],
    # 'EPOCHS': [200],
    # 'LEARNING_RATE': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    # 'GEOM_SCALE': [1e0, 1e-1, 1e-2, 1e-3],
    # 'RECURRENT_DROPOUT': [0.0],
    # 'PATIENCE': [0, 1, 4, 8, 16, 32],
    'EARLY_STOPPING': [1],
}
grid = list(ParameterGrid(HYPERPARAMS))

for configuration in grid:
    envs = []
    # Set environment variables (this allows you to do hyperparam searches from any scripting environment)
    for key, value in configuration.items():
        os.environ[key] = str(value)

    # repeat to get a sense of results spread
    for _ in range(N_TIMES):
        r_code = os.system('python3 neighborhood_inhabitants.py')
        if not r_code == 0:
            print('Neighborhood inhabitants grid search exited with error')
            notify('Neighborhood inhabitants grid search', 'with error')
            sys.exit(1)

signature = 'Neighborhood inhabitants grid search {} on {}'.format(SCRIPT_VERSION, socket.gethostname())
notify(signature, 'success')
print('Neighborhood inhabitants grid search finished successfully')
