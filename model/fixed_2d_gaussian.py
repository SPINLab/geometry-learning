from datetime import datetime
import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import Dense, LSTM, LeakyReLU
from keras.optimizers import Adam

from topoml_util.CustomCallback import CustomCallback
from topoml_util.geom_loss import r3_bivariate_gaussian_loss

TIMESTAMP = str(datetime.now()).replace(':', '.')
EPOCHS = 20
BATCH_SIZE = 512
TRAINING_SIZE = 100000
TRAIN_VALIDATE_SPLIT = 0.2
tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, histogram_freq=1, write_graph=True)

input_2d = np.repeat([[[0.2, 15, 0, 0, 0]]], 11, axis=1)
input_2d = np.repeat(input_2d, TRAINING_SIZE, axis=0)
(data_points, max_points, vector_len) = input_2d.shape

inputs = Input(name='Input', shape=(max_points, vector_len))
model = LSTM(vector_len, return_sequences=True)(inputs)
# The dense layer is required for input values exceeding 1e0
model = Dense(vector_len)(model)
model = Model(inputs, model)
model.compile(loss=r3_bivariate_gaussian_loss, optimizer=Adam(lr=0.01))
model.summary()

my_callback = CustomCallback(lambda x: str(x))

model.fit(x=input_2d,
          y=input_2d,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_split=TRAIN_VALIDATE_SPLIT,
          callbacks=[my_callback, tb_callback])
