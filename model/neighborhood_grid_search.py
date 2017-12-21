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
    for key, value in configuration.items():
        os.environ[key] = str(value)
    run(['python3', 'neighborhood_inhabitants.py'], stdout=PIPE)
