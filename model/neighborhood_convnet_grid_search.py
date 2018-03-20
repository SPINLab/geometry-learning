import os
import socket

import sys

from sklearn.model_selection import ParameterGrid
from topoml_util.slack_send import notify

SCRIPT_VERSION = '0.1.16'
SCRIPT_NAME = os.path.basename(__file__)
N_TIMES = 10

HYPERPARAMS = {
    # 'BATCH_SIZE': [512],
    # 'KERNEL_SIZE': np.linspace(64, 128, 3, dtype=int),
    # 'DENSE_SIZE': [64],
    # 'EPOCHS': [200],
    'LEARNING_RATE': [2e-3, 1e-3, 8e-4, 6e-4, 4e-4, 2e-4],
    # 'GEOM_SCALE': [1e0, 1e-1, 1e-2, 1e-3],
    # 'PATIENCE': [0, 1, 4, 8, 16, 32],
    # 'EARLY_STOPPING': [1],
}
grid = list(ParameterGrid(HYPERPARAMS))

for configuration in grid:
    envs = []
    # Set environment variables (this allows you to do hyperparam searches from any scripting environment)
    for key, value in configuration.items():
        os.environ[key] = str(value)

    # repeat to get a sense of results spread
    for _ in range(N_TIMES):
        r_code = os.system('python3 neighborhood_inhabitants_convnet.py')
        if not r_code == 0:
            print('Neighborhood inhabitants grid search exited with error')
            notify('Neighborhood inhabitants grid search', 'with error')
            sys.exit(1)

signature = '{} {} on {}'.format(SCRIPT_NAME, SCRIPT_VERSION, socket.gethostname())
notify(signature, 'success')
print('{} finished successfully'.format(SCRIPT_NAME))
