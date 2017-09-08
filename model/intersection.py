from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense, Flatten
from keras.optimizers import Adam, sgd

from topoml_util.CustomCallback import CustomCallback
from topoml_util.GeoVectorizer import GeoVectorizer
from topoml_util.GeoVectorizer import GEO_VECTOR_LEN as TARGET_GEO_VECTOR_LEN
from topoml_util.geom_loss import geom_gaussian_loss
from topoml_util.geom_scaler import localized_normal

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

TIMESTAMP = str(datetime.now())
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 512
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 128
EPOCHS = 50
OPTIMIZER = Adam(lr=1e-4)

loaded = np.load(DATA_FILE)
training_vectors = loaded['input_geoms']
training_vectors = localized_normal(training_vectors, 1e4)
target_vectors = loaded['intersection']
target_vectors = localized_normal(target_vectors, 1e4)
(_, max_points, GEO_VECTOR_LEN) = training_vectors.shape

inputs = Input(shape=(max_points, GEO_VECTOR_LEN))
model = LSTM(LATENT_SIZE, activation='relu', return_sequences=True)(inputs)
model = Dense(GEO_VECTOR_LEN)(model)
model = Model(inputs, model)
model.compile(loss=geom_gaussian_loss, optimizer=OPTIMIZER)
model.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, histogram_freq=1, write_graph=True)
my_callback = CustomCallback(GeoVectorizer.decypher)

history = model.fit(
    x=training_vectors,
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=[my_callback, tb_callback]
).history

print(history)
