from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

from topoml_util.geom_scaler import localized_normal
from topoml_util.CustomCallback import CustomCallback
from topoml_util.geom_loss import gaussian_1d_loss

# TODO: use recurrent dropout

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

TIMESTAMP = str(datetime.now()).replace(':', '.')
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 512
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 64
EPOCHS = 50
OPTIMIZER = Adam(lr=0.001)

loaded = np.load(DATA_FILE)
training_vectors = loaded['input_geoms']
(data_points, max_points, GEO_VECTOR_LEN) = training_vectors.shape

# Bring coordinates and distance in roughly the same scale
training_vectors = localized_normal(training_vectors, 1e4)
target_vectors = loaded['centroid_distance'][:, 0, :]

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
model = LSTM(LATENT_SIZE, activation='relu')(inputs)
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
