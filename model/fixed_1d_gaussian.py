from datetime import datetime
import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

from topoml_util.CustomCallback import CustomCallback
from topoml_util.geom_loss import gaussian_1d_loss

TIMESTAMP = str(datetime.now())
TIMESTAMP = TIMESTAMP.replace(':', '.')
EPOCHS = 60
BATCH_SIZE = 100
TRAINING_SIZE = 100000
TRAIN_VALIDATE_SPLIT = 0.2
tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, histogram_freq=1, write_graph=True)

input_1d = np.repeat([[[52]]], TRAINING_SIZE, axis=0)
input_1d = np.append(input_1d, np.zeros(shape=(TRAINING_SIZE, 1, 1)), axis=2)
(_, max_points, GEO_VECTOR_LEN) = input_1d.shape

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
model = LSTM(GEO_VECTOR_LEN, return_sequences=True)(inputs)
model = TimeDistributed(Dense(GEO_VECTOR_LEN))(model)
model = Model(inputs, model)
model.compile(loss=gaussian_1d_loss, optimizer=Adam(lr=0.01))
model.summary()

my_callback = CustomCallback(lambda x: str(x))

model.fit(x=input_1d,
          y=input_1d,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_split=TRAIN_VALIDATE_SPLIT,
          callbacks=[my_callback, tb_callback])
