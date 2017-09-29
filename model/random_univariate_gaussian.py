from datetime import datetime
import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

from topoml_util.ConsoleLogger import DecypherAll
from topoml_util.geom_loss import r3_univariate_gaussian_loss

TIMESTAMP = str(datetime.now()).replace(':', '.')
EPOCHS = 60
BATCH_SIZE = 100
TRAINING_SIZE = 50000
TRAIN_VALIDATE_SPLIT = 0.2
tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, histogram_freq=1, write_graph=True)

input_1d = np.random.randint(low=1, high=20, size=(TRAINING_SIZE, 1, 1))
input_1d = np.append(input_1d, np.zeros(shape=(TRAINING_SIZE, 1, 1)), axis=2)
(_, max_points, GEO_VECTOR_LEN) = input_1d.shape

inputs = Input(name='Input', shape=(max_points, GEO_VECTOR_LEN))
model = LSTM(GEO_VECTOR_LEN, return_sequences=True)(inputs)
model = TimeDistributed(Dense(GEO_VECTOR_LEN))(model)
model = Model(inputs, model)
model.compile(loss=r3_univariate_gaussian_loss, optimizer=Adam(lr=0.001))
model.summary()

my_callback = DecypherAll(lambda x: str(x))

model.fit(x=input_1d,
          y=input_1d,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_split=TRAIN_VALIDATE_SPLIT,
          callbacks=[my_callback, tb_callback])