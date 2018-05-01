import os
import socket

import sys

# import numpy as np
from sklearn.model_selection import ParameterGrid
from topoml_util.slack_send import notify

SCRIPT_NAME = os.path.basename(__file__)
SCRIPT_VERSION = '1.0.1'
SIGNATURE = '{} {} on {}'.format(SCRIPT_NAME, SCRIPT_VERSION, socket.gethostname())
N_TIMES = 1

if len(sys.argv) > 1:
    script_name = sys.argv[1]
else:  # resort to default, for
    # script_name = 'neighborhood_dense.py'
    # script_name = 'neighborhood_convnet.py'
    # script_name = 'neighborhood_lstm.py'
    # script_name = 'building_dense.py'
    # script_name = 'building_convnet.py'
    # script_name = 'building_lstm.py'
    # script_name = 'archaeology_dense.py'
    script_name = 'archaeology_convnet.py'
    # script_name = 'archaeology_lstm.py'

HYPERPARAMS = {
    # 'BATCH_SIZE': [512],
    # 'REPEAT_DEEP_ARCH': [1, 2],
    # 'KERNEL_SIZE': np.linspace(1, 8, 8, dtype=int),
    # 'LSTM_SIZE': np.linspace(64, 128, 3, dtype=int),
    # 'DENSE_SIZE': [64],
    # 'EPOCHS': [200],
    # 'LEARNING_RATE': [8e-4, 6e-4, 4e-4, 2e-4, 1e-4],
    'LEARNING_RATE': [1e-2, 8e-3, 6e-3, 4e-3, 2e-3],
    # 'GEOM_SCALE': [1e0, 1e-1, 1e-2, 1e-3],
    # 'RECURRENT_DROPOUT': [0.0],
    # 'PATIENCE': [0, 1, 4, 8, 16, 32],
    # 'EARLY_STOPPING': [0],
}
grid = list(ParameterGrid(HYPERPARAMS))

for configuration in grid:
    envs = []
    # Set environment variables (this allows you to do hyperparam searches from any scripting environment)
    for key, value in configuration.items():
        os.environ[key] = str(value)

    # repeat to get a sense of results spread
    for _ in range(N_TIMES):
        r_code = os.system('python3 {}'.format(script_name))
        if not r_code == 0:
            print('Grid search exited with error')
            notify(SIGNATURE, 'error')
            sys.exit(1)

notify(SIGNATURE, 'success')
print('Grid search {} finished successfully'.format(SIGNATURE))
