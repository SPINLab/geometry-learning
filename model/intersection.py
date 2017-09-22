from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam, sgd

from topoml_util.ConsoleLogger import DecypherAll
from topoml_util.GeoVectorizer import GeoVectorizer, FULL_STOP_INDEX
from topoml_util.GaussianMixtureLoss import GaussianMixtureLoss
from topoml_util.geom_scaler import localized_normal, localized_mean

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

TIMESTAMP = str(datetime.now()).replace(':', '.')
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 512
GAUSSIAN_MIXTURE_COMPONENTS = 5
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 128
EPOCHS = 50
OPTIMIZER = Adam(lr=1e-4)

lossfunc = GaussianMixtureLoss(GAUSSIAN_MIXTURE_COMPONENTS)

loaded = np.load(DATA_FILE)
raw_training_vectors = loaded['input_geoms']
raw_target_vectors = loaded['intersection']

training_vectors = []
target_vectors = []

# skip non-intersecting geometries
for train, target in zip(raw_training_vectors, raw_target_vectors):
    if not target[0, 0] == 0:  # a zero coordinate designates an empty geometry
        training_vectors.append(train)
        target_vectors.append(target)

means = localized_mean(training_vectors)
training_vectors = localized_normal(training_vectors, means, 1e4)
target_vectors = localized_normal(target_vectors, means, 1e4)
(data_points, max_points, INPUT_VECTOR_LEN) = training_vectors.shape

component_1 = target_vectors[:, :, 0:5]  # The first gaussian mixture model component for each point
component_1 = np.append(component_1, np.zeros((data_points, max_points, 1)), axis=2)  # Add pi feature
target_vectors = np.append(
    np.reshape(
        np.repeat(component_1, GAUSSIAN_MIXTURE_COMPONENTS, axis=1),
        (data_points, max_points, 6 * GAUSSIAN_MIXTURE_COMPONENTS)),
    target_vectors[:, :, 5:], axis=2)
(_, _, OUTPUT_VECTOR_LEN) = target_vectors.shape

inputs = Input(shape=(max_points, INPUT_VECTOR_LEN))
model = LSTM(LATENT_SIZE, activation='relu', return_sequences=True)(inputs)
model = TimeDistributed(Dense(64))(model)
model = Dense(OUTPUT_VECTOR_LEN)(model)
model = Model(inputs, model)
model.compile(loss=lossfunc.geom_gaussian_mixture_loss, optimizer=OPTIMIZER)
model.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, histogram_freq=1, write_graph=True)
my_callback = DecypherAll(lambda x: str(x))

history = model.fit(
    x=training_vectors,
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=[my_callback, tb_callback]
).history

print(history)
