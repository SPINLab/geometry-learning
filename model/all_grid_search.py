import os

import sys
from sklearn.model_selection import ParameterGrid
from topoml_util.slack_send import notify

SCRIPT_VERSION = '0.0.7'
N_TIMES = 10

HYPERPARAMS = {
    'BATCH_SIZE': [512],
    'REPEAT_DEEP_ARCH': [0],
    'LSTM_SIZE': [64],
    'DENSE_SIZE': [32],
    'EPOCHS': [200],
    'LEARNING_RATE': [1e-4],
    # 'GEOM_SCALE': [1e0, 1e-1, 1e-2, 1e-3],
    'RECURRENT_DROPOUT': [0.10],
    # 'PATIENCE': [8, 16, 24, 32, 40],
}
grid = list(ParameterGrid(HYPERPARAMS))

scripts = [
    'neighborhood_inhabitants.py',
    'building_type.py',
    'archaeological_features.py'
]

for configuration in grid:
    # Set environment variables (this allows you to do hyperparam searches from any scripting environment)
    for key, value in configuration.items():
        os.environ[key] = str(value)

    # repeat to get a sense of results spread
    for _ in range(N_TIMES):
        for script in scripts:
            r_code = os.system('python3 {}'.format(script))
            if not r_code == 0:
                print('{} exited with error'.format(script))
                notify('{} grid search'.format(script), 'with error')
                sys.exit(1)

notify('All grid search', 'no errors')
print('All grid search', 'finished successfully')
