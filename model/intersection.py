from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam, sgd

from topoml_util.CustomCallback import CustomCallback
from topoml_util.GeoVectorizer import GeoVectorizer
from topoml_util.GeoVectorizer import GEO_VECTOR_LEN as TARGET_GEO_VECTOR_LEN
from topoml_util.geom_loss import geom_gaussian_loss

# TODO: increase the num_steps in the training set to 10,000,000 (like sketch-rnn)
# TODO: use recurrent dropout

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

TIMESTAMP = str(datetime.now())
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 100
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 256
EPOCHS = 500
OPTIMIZER = 'adam'

loaded = np.load(DATA_FILE)
training_vectors = loaded['input_geoms']
target_vectors = loaded['intersection']
(_, max_points, GEO_VECTOR_LEN) = training_vectors.shape

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
encoded = LSTM(GEO_VECTOR_LEN, return_sequences=True)(inputs)
encoded = TimeDistributed(Dense(LATENT_SIZE))(encoded)
encoder = Model(inputs, encoded)
encoder.summary()

encoded_inputs = Input(shape=(max_points, LATENT_SIZE))
decoded = LSTM(LATENT_SIZE, return_sequences=True,
               bias_initializer='ones',
               kernel_initializer='random_uniform')(encoded_inputs)
decoded = TimeDistributed(Dense(TARGET_GEO_VECTOR_LEN))(decoded)
decoder = Model(encoded_inputs, decoded)

ae = Model(inputs, decoder(encoder(inputs)))
ae.compile(loss=geom_gaussian_loss, optimizer=sgd(lr=0.0001))
ae.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, histogram_freq=1, write_graph=True)
my_callback = CustomCallback(GeoVectorizer.decypher)

history = ae.fit(x=training_vectors,
                 y=target_vectors,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 validation_split=TRAIN_VALIDATE_SPLIT,
                 callbacks=[my_callback, tb_callback]).history

print(history)
