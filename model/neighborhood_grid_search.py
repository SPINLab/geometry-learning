from subprocess import run, PIPE

import os
from sklearn.model_selection import ParameterGrid

SCRIPT_VERSION = '0.0.2'
HYPERPARAMS = {
    'BATCH_SIZE': [32, 64, 128],
    'REPEAT_DEEP_ARCH': [0, 2, 4],
    'DENSE_SIZE': [16, 64, 256],
    'LSTM_SIZE': [64, 128, 256],
    'LEARNING_RATE': [1e-4, 1e-3, 1e-2]
}

grid = list(ParameterGrid(HYPERPARAMS))

for configuration in grid:
    envs = []
    for key, value in configuration.items():
        os.environ[key] = str(value)
    run(['python3', 'neighborhood_inhabitants.py'], stdout=PIPE)
