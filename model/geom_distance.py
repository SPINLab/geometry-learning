import os
from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from topoml_util.ConsoleLogger import DecypherAll
from topoml_util.gaussian_loss import univariate_gaussian_loss
from topoml_util.geom_scaler import localized_normal, localized_mean
from topoml_util.slack_send import notify

SCRIPT_VERSION = "0.1.3"
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
PLOT_DIR = 'plots/' + SIGNATURE
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 512
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 256
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-3)

loaded = np.load(DATA_FILE)
raw_input_vectors = loaded['input_geoms']

# Bring coordinates and distance in the same scale
means = localized_mean(raw_input_vectors)
raw_input_vectors = localized_normal(raw_input_vectors, means, 1e4)

(data_points, max_points, GEO_VECTOR_LEN) = raw_input_vectors.shape
raw_target_vectors = loaded['geom_distance'][:, 0, :]

input_vectors = []
target_vectors = []
intersecting_count = 0
non_intersecting_count = 0

# create 50/50 intersecting/non-intersecting distribution
for inputs, targets in zip(raw_input_vectors, raw_target_vectors):
    if targets[0] == 0 and non_intersecting_count < intersecting_count:
        input_vectors.append(inputs)
        target_vectors.append(targets)
        non_intersecting_count += 1
    elif targets[0] != 0:
        input_vectors.append(inputs)
        target_vectors.append(targets)
        intersecting_count += 1

input_vectors = np.array(input_vectors)
target_vectors = np.array(target_vectors)

# plot distance distribution
mu = np.mean(target_vectors[:, 0])
sigma = np.std(target_vectors[:, 0])
fig, ax = plt.subplots()
plt.text(0.70, 0.94, r'distance $\mu: $' + str(np.round(mu, 4)), transform=ax.transAxes)
plt.text(0.70, 0.88, r'distance $\sigma: $' + str(np.round(sigma, 4)), transform=ax.transAxes)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Geometric distance distribution in meters')
n, bins, patches = plt.hist(target_vectors[:, 0], 50, facecolor='g', normed=False, log=True, alpha=0.75)
os.makedirs(str(PLOT_DIR), exist_ok=True)
plt.savefig(PLOT_DIR + '/plt_' + SIGNATURE + '_distance_distr.png')

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
model = LSTM(LATENT_SIZE, activation='relu')(inputs)
model = Dense(2, activation='relu')(model)
model = Model(inputs, model)
model.compile(
    loss=univariate_gaussian_loss,
    optimizer=OPTIMIZER)
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP + ' ' + SCRIPT_NAME, write_graph=False),
    DecypherAll(lambda x: str(x)),
    EarlyStopping(patience=40, min_delta=0.001)
]

history = model.fit(
    x=input_vectors,
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

val_set_start = -round(data_points * TRAIN_VALIDATE_SPLIT)
prediction = model.predict(raw_input_vectors[val_set_start:])
error = prediction[:, 0] - raw_target_vectors[val_set_start:, 0]
_, ax = plt.subplots()
plt.text(0.01, 0.94, r'prediction error $\mu: $' + str(np.round(np.mean(error), 4)), transform=ax.transAxes)
plt.text(0.01, 0.88, r'prediction error $\sigma: $' + str(np.round(np.std(error), 4)), transform=ax.transAxes)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Geometric distance error distribution in meters')
plt.hist(error, 50, facecolor='g', normed=False, alpha=0.75)
os.makedirs(str(PLOT_DIR), exist_ok=True)
plt.savefig(PLOT_DIR + '/plt_' + SIGNATURE + '.png')

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
