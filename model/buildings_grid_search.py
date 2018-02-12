import os

import sys
from sklearn.model_selection import ParameterGrid
from topoml_util.slack_send import notify

SCRIPT_VERSION = '0.0.3'
N_TIMES = 10

HYPERPARAMS = {
    # 'BATCH_SIZE': [512],
    # 'REPEAT_DEEP_ARCH': [0, 1],
    # 'LSTM_SIZE': [192, 256, 384],
    # 'DENSE_SIZE': [64],
    # 'EPOCHS': [200],
    # 'LEARNING_RATE': [1e-3, 3e-4, 1e-4],
    # 'GEOM_SCALE': [1e0, 1e-1, 1e-2, 1e-3],
    # 'RECURRENT_DROPOUT': [0.0, 0.05, 0.1],
    'PATIENCE': [24, 32, 40, 48],
}
grid = list(ParameterGrid(HYPERPARAMS))

for configuration in grid:
    envs = []
    # Set environment variables (this allows you to do hyperparam searches from any scripting environment)
    for key, value in configuration.items():
        os.environ[key] = str(value)

    # repeat to get a sense of results spread
    for _ in range(N_TIMES):
        r_code = os.system('python3 building_type.py')
        if not r_code == 0:
            print('Building type grid search exited with error')
            notify('Building type grid search', 'with error')
            sys.exit(1)

notify('Building type grid search', 'no errors')
print('Building type grid search', 'finished successfully')
