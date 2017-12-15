import os

from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense, Flatten
from model.topoml_util.geom_scaler import localized_mean, localized_normal
from keras.optimizers import Adam
from slack_send import notify

SCRIPT_VERSION = '0.0.2'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
TRAINING_DATA_FILE = '../files/neighborhoods/neighborhoods_train.npz'
BATCH_SIZE = 1024
GAUSSIAN_MIXTURE_COMPONENTS = 1
TRAIN_VALIDATE_SPLIT = 0.1
REPEAT_DEEP_ARCH = 0
LSTM_SIZE = 128
DENSE_SIZE = 64
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-4, clipnorm=1.)

train_loaded = np.load(TRAINING_DATA_FILE)
train_geoms = train_loaded['input_geoms']
train_above_or_below_median = train_loaded['above_or_below_median']

# Normalize
means = localized_mean(train_geoms)
variance = np.var(train_geoms[..., 0:2])
train_geoms = localized_normal(train_geoms, means, variance)

# Shape determination
geom_max_points, geom_vector_len = train_geoms.shape[1:]
output_seq_length = train_above_or_below_median.shape[-1]

# Build model
inputs = Input(shape=(geom_max_points, geom_vector_len))
model = LSTM(geom_max_points, activation='relu', return_sequences=True)(inputs)

for layer in range(REPEAT_DEEP_ARCH):
    model = LSTM(LSTM_SIZE, activation='relu', return_sequences=True)(model)

model = TimeDistributed(Dense(DENSE_SIZE, activation='relu'))(model)
model = Flatten()(model)
model = Dense(output_seq_length)(model)

model = Model(inputs=inputs, outputs=model)
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=OPTIMIZER),
model.summary()

# Callbacks
callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + SIGNATURE, write_graph=False),
    EarlyStopping(patience=40, min_delta=1)
]

history = model.fit(
    x=train_geoms,
    y=train_above_or_below_median,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

notify(TIMESTAMP, SCRIPT_NAME, 'accuracy of ' + str(history['accuracy'][-1]))
print(SCRIPT_NAME, 'finished successfully')
