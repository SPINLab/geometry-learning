from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from topoml_util.GeoVectorizer import GeoVectorizer
from topoml_util.LoggerCallback import EpochLogger

TIMESTAMP = str(datetime.now()).replace(':', '.')
DATA_FILE = '../files/triangles.npz'
BATCH_SIZE = 8192
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 18
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-3)

loaded = np.load(DATA_FILE)
training_vectors = loaded['point_sequence']
(set_size, GEO_VECTOR_LEN) = training_vectors.shape
training_vectors = np.reshape(training_vectors, (set_size, 1, GEO_VECTOR_LEN))
target_vectors = loaded['intersection_surface']

inputs = Input(shape=(1, GEO_VECTOR_LEN))
model = LSTM(64, activation='relu', return_sequences=True)(inputs)
model = Dense(32, activation='relu')(model)
model = LSTM(64, activation='relu', return_sequences=True)(model)
model = Dense(32, activation='relu')(model)
model = LSTM(64, activation='relu')(model)
model = Dense(1)(model)
model = Model(inputs, model)
model.compile(loss='mse', optimizer=OPTIMIZER)
model.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, histogram_freq=1, write_graph=True)
epoch_callback = EpochLogger(
    input_func=GeoVectorizer.decypher,
    target_func=lambda x: str(x),
    predict_func=lambda x: str(x),
    aggregate_func=None,
    stdout=True
)

model.fit(
    x=training_vectors,
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=[epoch_callback, tb_callback]
)
