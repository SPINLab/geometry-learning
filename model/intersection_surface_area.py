import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense, concatenate, Reshape, LeakyReLU
from keras.optimizers import Adam
from topoml_util.LoggerCallback import EpochLogger
from topoml_util.gaussian_loss import univariate_gaussian_loss
from topoml_util.geom_scaler import localized_normal, localized_mean
from topoml_util.slack_send import notify

SCRIPT_VERSION = "0.0.17"
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
DATA_FILE = '../files/geodata_vectorized.npz'
PLOT_DIR = 'plots/' + SIGNATURE
BATCH_SIZE = 1024
TRAIN_VALIDATE_SPLIT = 0.1
LSTM_UNITS = 256
DENSE_UNITS = 64
REPEAT_HIDDEN = 1
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-4)

loaded = np.load(DATA_FILE)
raw_brt_vectors = loaded['brt_vectors']
raw_osm_vectors = loaded['osm_vectors']
raw_surface_area_vectors = loaded['intersection_surface'][:, 0, 0]
raw_combined_geom_vectors = loaded['input_geoms']

brt_vectors = []
osm_vectors = []
area_vectors = []
combined_geom_vectors = []
intersecting_count = 0
non_intersecting_count = 0

# create 50/50 intersecting/non-intersecting distribution
for brt, osm, surface, combined in zip(raw_brt_vectors, raw_osm_vectors, raw_surface_area_vectors,
                                       raw_combined_geom_vectors):
    if surface > 0 and non_intersecting_count > intersecting_count:
        brt_vectors.append(brt)
        osm_vectors.append(osm)
        area_vectors.append(surface)
        combined_geom_vectors.append(combined)
        intersecting_count += 1
    elif surface == 0:
        brt_vectors.append(brt)
        osm_vectors.append(osm)
        area_vectors.append(surface)
        combined_geom_vectors.append(combined)
        non_intersecting_count += 1

# data whitening
means = localized_mean(combined_geom_vectors)
brt_vectors = localized_normal(brt_vectors, means, 1e4)
osm_vectors = localized_normal(osm_vectors, means, 1e4)
area_vectors = np.array(area_vectors)

# shape determination
data_points, brt_max_points, brt_seq_len = brt_vectors.shape
_, osm_max_points, osm_seq_len = osm_vectors.shape

# plot target surface area distribution
mu = np.mean(area_vectors)
sigma = np.std(area_vectors)
fig, ax = plt.subplots()
plt.text(0.70, 0.94, r'area $\mu: $' + str(np.round(mu, 4)), transform=ax.transAxes)
plt.text(0.70, 0.88, r'area $\sigma: $' + str(np.round(sigma, 4)), transform=ax.transAxes)
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.title('Intersection surface area distribution')
n, bins, patches = plt.hist(area_vectors, bins=50, facecolor='g',
                            log=True, alpha=0.75)
os.makedirs(str(PLOT_DIR), exist_ok=True)
plt.savefig(PLOT_DIR + '/plt_' + SIGNATURE + '_area_distr.png')

# Build model
brt_inputs = Input(shape=(brt_max_points, brt_seq_len))
brt_model = LSTM(brt_max_points * 2, activation='relu')(brt_inputs)

osm_inputs = Input(shape=(osm_max_points, osm_seq_len))
osm_model = LSTM(osm_max_points * 2, activation='relu')(osm_inputs)

concat = concatenate([brt_model, osm_model])
model = Reshape((1, concat.shape[-1].value))(concat)

for layer in range(REPEAT_HIDDEN):
    model = LSTM(LSTM_UNITS, activation='relu', return_sequences=True)(model)
    model = Dense(DENSE_UNITS, activation='relu')(model)

model = LSTM(LSTM_UNITS, activation='relu')(model)  # Flatten
model = Dense(2)(model)
model = Model(inputs=[brt_inputs, osm_inputs], outputs=model)
model.compile(loss=univariate_gaussian_loss, optimizer=OPTIMIZER)
model.summary()

callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + SIGNATURE, write_graph=False),
    EpochLogger(input_slice=lambda x: x[0:2], target_slice=lambda x: x[2:3], stdout=True),
    EarlyStopping(patience=40, min_delta=0.001)
]

history = model.fit(
    x=[brt_vectors, osm_vectors],
    y=area_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

val_set_start = -round(data_points * TRAIN_VALIDATE_SPLIT)
prediction = model.predict([brt_vectors[val_set_start:], osm_vectors[val_set_start:]])
error = prediction[:, 0] - area_vectors[val_set_start:]
_, ax = plt.subplots()
plt.text(0.01, 0.94, r'prediction error $\mu: $' + str(np.round(np.mean(error), 4)), transform=ax.transAxes)
plt.text(0.01, 0.88, r'prediction error $\sigma: $' + str(np.round(np.std(error), 4)), transform=ax.transAxes)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Intersection area error distribution')
plt.hist(error, 50, facecolor='g', normed=False, alpha=0.75)
os.makedirs(str(PLOT_DIR), exist_ok=True)
plt.savefig(PLOT_DIR + '/plt_' + SIGNATURE + '_error_distr.png')

notify(TIMESTAMP, SCRIPT_NAME, 'validation loss of ' + str(history['val_loss'][-1]))
print(SCRIPT_NAME, 'finished successfully')
