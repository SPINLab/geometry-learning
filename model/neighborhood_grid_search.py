from subprocess import run, PIPE

import os
from sklearn.model_selection import ParameterGrid

SCRIPT_VERSION = '0.0.3'
HYPERPARAMS = {
    'BATCH_SIZE': [8, 16],
    'REPEAT_DEEP_ARCH': [0],
    'DENSE_SIZE': [16, 64],
    'LSTM_SIZE': [64, 128],
    'EPOCHS': [10, 20],
    'LEARNING_RATE': [1e-4, 1e-3, 1e-2]
}

grid = list(ParameterGrid(HYPERPARAMS))

for configuration in grid:
    envs = []
    # Set environment variables (this allows you to do hyperparam searches from any scripting environment)
    for key, value in configuration.items():
        os.environ[key] = str(value)
    os.system('neighborhood_inhabitants.py')
