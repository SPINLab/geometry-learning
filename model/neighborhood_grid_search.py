from subprocess import run

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
    envs = ' '.join(['{}={}'.format(key, str(value)) for key, value in configuration.items()])
    run('{} python3 neighborhood_inhabitants.py'.format(envs))
