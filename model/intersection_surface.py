from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

from topoml_util.CustomCallback import CustomCallback
from topoml_util.geom_loss import gaussian_1d_loss
from topoml_util.geom_scaler import localized_normal, localized_mean

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

TIMESTAMP = str(datetime.now())
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 512
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 128
EPOCHS = 50
OPTIMIZER = Adam(lr=1e-4)

loaded = np.load(DATA_FILE)
raw_training_vectors = loaded['input_geoms']
raw_target_vectors = loaded['intersection_surface'][:, 0, :]

training_vectors = []
target_vectors = []

for index, target in enumerate(raw_target_vectors):
    if not target[0] == 0:  # a zero coordinate designates an empty geometry
        training_vectors.append(raw_training_vectors[index])
        target_vectors.append(raw_target_vectors[index])

means = localized_mean(training_vectors)
training_vectors = localized_normal(training_vectors, means, 1e4)
target_vectors = np.array(target_vectors)
(_, max_points, GEO_VECTOR_LEN) = training_vectors.shape

inputs = Input(shape=(max_points, GEO_VECTOR_LEN))
model = Dense(128, activation='relu')(inputs)
model = LSTM(LATENT_SIZE, activation='relu')(model)
model = Dense(2)(model)
model = Model(inputs, model)
model.compile(loss=gaussian_1d_loss, optimizer=OPTIMIZER)
model.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, histogram_freq=1, write_graph=True)
my_callback = CustomCallback(lambda x: str(x))

history = model.fit(
    x=training_vectors,
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=[my_callback, tb_callback]
).history

print(history)

prediction = model.predict(training_vectors[0:1000])
intersecting_error = np.abs(prediction[:, 0] - target_vectors[0:1000, 0])
print('Intersection surface area mean:', np.mean(target_vectors))
print('Intersecting error mean:', np.mean(intersecting_error))
