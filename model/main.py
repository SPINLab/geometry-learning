import numpy as np
from keras import Input, losses
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM

from topoml_util.CustomCallback import CustomCallback
from topoml_util.GeoVectorizer import GeoVectorizer
from topoml_util.geom_loss import geom_loss

# TODO: increase the num_steps in the training set to 10,000,000 (like sketch-rnn)
# TODO: fiddle with the batch size on CUDA cores
# TODO: use recurrent dropout

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 1000
TRAIN_VALIDATE_SPLIT = 0.2
RNN_WIDTH = 20
EPOCHS = 300

loaded = np.load(DATA_FILE)
training_vectors = loaded['X']
target_vectors = loaded['y']
(_, max_points, GEO_VECTOR_LEN) = training_vectors.shape

inputs = Input(name="Input", shape=(max_points, GEO_VECTOR_LEN))
encoded = LSTM(GEO_VECTOR_LEN, return_sequences=True)(inputs)
encoded = LSTM(GEO_VECTOR_LEN, return_sequences=True)(encoded)
encoded = LSTM(GEO_VECTOR_LEN, return_sequences=True)(encoded)
encoder = Model(inputs, encoded)
encoder.summary()

encoded_inputs = Input(shape=(max_points, GEO_VECTOR_LEN))
decoded = LSTM(GEO_VECTOR_LEN, return_sequences=True)(encoded_inputs)
decoded = LSTM(GEO_VECTOR_LEN, return_sequences=True)(decoded)
decoded = LSTM(GEO_VECTOR_LEN, return_sequences=True)(decoded)
decoder = Model(encoded_inputs, decoded)

ae = Model(inputs, decoder(encoder(inputs)))
ae.compile(loss=geom_loss, optimizer='rmsprop')
ae.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log', histogram_freq=1, write_graph=True)
my_callback = CustomCallback(GeoVectorizer.decypher)

ae.fit(x=training_vectors,
       y=target_vectors,
       epochs=EPOCHS,
       batch_size=BATCH_SIZE,
       validation_split=TRAIN_VALIDATE_SPLIT,
       callbacks=[my_callback, tb_callback])
