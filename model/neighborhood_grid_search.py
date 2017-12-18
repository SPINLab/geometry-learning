from subprocess import run, PIPE

import os
from sklearn.model_selection import ParameterGrid

hyperparams = {
    'BATCH_SIZE': [32, 64, 256, 512],
    'REPEAT_DEEP_ARCH': [0, 2, 4],
    'DENSE_SIZE': [16, 64, 256],
    'LSTM_SIZE': [64, 128, 256],
    'LEARNING_RATE': [1e-4, 1e-3, 1e-2]
}

grid = list(ParameterGrid(hyperparams))

for configuration in grid:
    envs = []
    for key, value in configuration.items():
        os.environ[key] = '"{}"'.format(value)
    run(['python3', 'neighborhood_inhabitants.py'], stdout=PIPE)
