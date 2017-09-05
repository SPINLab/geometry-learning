from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from topoml_util.CustomCallback import CustomCallback
from topoml_util.geom_loss import gaussian_1d_loss

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

TIMESTAMP = str(datetime.now()).replace(':', '.')
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 512
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 16
EPOCHS = 50
OPTIMIZER = Adam(lr=0.001)
# OPTIMIZER = SGD

loaded = np.load(DATA_FILE)
training_vectors = loaded['centroids'][:, :, 0:2]

# Bring coordinates and distance in roughly the same scale
base_precision = 1e4
base = np.floor(base_precision * training_vectors[:, 0:1, :])
base = np.repeat(base, 2, axis=1)
training_vectors = (base_precision * training_vectors) - base

(data_points, max_points, GEO_VECTOR_LEN) = training_vectors.shape
target_vectors = loaded['centroid_distance'][:, 0, :]

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
# This can be a simple Dense layer of size 16 as well
model = Flatten()(inputs)
model = Dense(LATENT_SIZE, activation='relu')(model)
model = Dense(2)(model)

model = Model(inputs, model)
model.compile(loss=gaussian_1d_loss, optimizer=OPTIMIZER)
model.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, histogram_freq=1, write_graph=True)
my_callback = CustomCallback(lambda x: str(x))

history = model.fit(x=training_vectors,
                    y=target_vectors,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=TRAIN_VALIDATE_SPLIT,
                    callbacks=[my_callback, tb_callback]).history

print(history)
