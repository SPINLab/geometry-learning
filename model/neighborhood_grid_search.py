from subprocess import run, PIPE

import os
from sklearn.model_selection import ParameterGrid
from slack_send import notify

SCRIPT_VERSION = '0.0.4'

HYPERPARAMS = {
    'BATCH_SIZE': [8, 16],
    'REPEAT_DEEP_ARCH': [0, 1],
    'LSTM_SIZE': [64, 128],
    'DENSE_SIZE': [16, 64],
    'EPOCHS': [20],
    'LEARNING_RATE': [1e-4, 3e-4, 1e-3]
}
grid = list(ParameterGrid(HYPERPARAMS))

for configuration in grid:
    envs = []
    # Set environment variables (this allows you to do hyperparam searches from any scripting environment)
    for key, value in configuration.items():
        os.environ[key] = str(value)
    os.system('neighborhood_inhabitants.py')

notify('Neighborhood inhabitants grid search', 'no errors')
print('Neighborhood inhabitants grid search', 'finished successfully')
