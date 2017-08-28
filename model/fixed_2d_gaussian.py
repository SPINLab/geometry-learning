from datetime import datetime
import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

from topoml_util.CustomCallback import CustomCallback
from topoml_util.geom_loss import gaussian_2d_loss

TIMESTAMP = str(datetime.now()).replace(':', '.')
EPOCHS = 100
BATCH_SIZE = 100
TRAINING_SIZE = 100000
TRAIN_VALIDATE_SPLIT = 0.2
tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, histogram_freq=1, write_graph=True)

input_2d_ones = np.ones(shape=(TRAINING_SIZE, 1, 2))
input_2d_zeros = np.zeros(shape=(TRAINING_SIZE, 1, 3))
input_2d = np.append(input_2d_ones, input_2d_zeros, axis=2)
(_, max_points, GEO_VECTOR_LEN) = input_2d.shape

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
model = LSTM(GEO_VECTOR_LEN, return_sequences=True)(inputs)
model = TimeDistributed(Dense(GEO_VECTOR_LEN))(model)
model = Model(inputs, model)
model.compile(loss=gaussian_2d_loss, optimizer=Adam(lr=0.0001))
model.summary()

my_callback = CustomCallback(lambda x: str(x))

model.fit(x=input_2d,

          y=input_2d,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_split=TRAIN_VALIDATE_SPLIT,
          callbacks=[my_callback, tb_callback])
