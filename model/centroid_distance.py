from datetime import datetime

import numpy as np
from keras import Input
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense, Reshape, Flatten, LeakyReLU, Lambda
from keras.optimizers import Adam, sgd

from topoml_util.CustomCallback import CustomCallback
from topoml_util.geom_loss import gaussian_1d_loss

# TODO: use recurrent dropout

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

TIMESTAMP = str(datetime.now()).replace(':', '.')
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 100
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 16
EPOCHS = 500
OPTIMIZER = Adam(lr=0.01)

loaded = np.load(DATA_FILE)
training_vectors = loaded['centroids_rd'][:, :, 0:2]
(data_points, max_points, GEO_VECTOR_LEN) = training_vectors.shape
target_vectors = np.sqrt(
    np.square(training_vectors[:, 0, 0] - training_vectors[0:, 1, 0]) +  # delta rd_x
    np.square(training_vectors[:, 0, 1] - training_vectors[0:, 1, 1]))   # delta rd_y
# target_vectors = np.repeat(target_vectors, max_points, axis=1)

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
model = Dense(16, activation='relu')(inputs)
model = LeakyReLU(alpha=0.3)(model)
model = Dense(1)(model)
model = Lambda(lambda x: K.mean(x), output_shape=(None,))(model)
model = Model(inputs, model)
model.compile(loss='mse', optimizer=OPTIMIZER)
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
