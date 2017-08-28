from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam, sgd

from topoml_util.CustomCallback import CustomCallback

# TODO: increase the num_steps in the training set to 10,000,000 (like sketch-rnn)
# TODO: use recurrent dropout

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2
from topoml_util.geom_loss import gaussian_1d_loss

TIMESTAMP = str(datetime.now()).replace(':', '.')
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 100
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 256
EPOCHS = 500
OPTIMIZER = 'adam'

loaded = np.load(DATA_FILE)
training_vectors = loaded['input_geoms']
(data_points, max_points, GEO_VECTOR_LEN) = training_vectors.shape
target_vectors = loaded['geom_distance']
target_vectors = np.repeat(target_vectors, max_points, axis=1)

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
model = LSTM(128, return_sequences=True)(inputs)
model = TimeDistributed(Dense(2))(model)
model = Model(inputs, model)
model.compile(loss=gaussian_1d_loss, optimizer=Adam(lr=0.001))
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
