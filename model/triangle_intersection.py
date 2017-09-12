from datetime import datetime

from shapely.wkt import loads
from shapely.geometry import Polygon
import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense, Flatten
from keras.optimizers import Adam, sgd

from topoml_util.CustomCallback import CustomCallback
from topoml_util.GeoVectorizer import GeoVectorizer
from topoml_util.geom_loss import geom_gaussian_loss

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

TIMESTAMP = str(datetime.now())
BATCH_SIZE = 1024
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 256
EPOCHS = 50
OPTIMIZER = Adam(lr=1e-4)

print('Creating triangles')
raw_training_vectors = np.random.normal(size=(100000, 6, 2))
triangle_sets = np.array([[Polygon(point_set[0:3]).wkt, Polygon(point_set[3:]).wkt]
                          for point_set in raw_training_vectors])
max_points = GeoVectorizer.max_points(triangle_sets[:, 0], triangle_sets[:, 1])
raw_training_vectors = [GeoVectorizer.vectorize_two_wkts(*triangle_set, max_points)
                        for triangle_set in triangle_sets]
# raw_training_vectors = [GeoVectorizer.interpolate(point_sequence, len(point_sequence) * 20)
#                         for point_sequence in raw_training_vectors]
(_, max_points, GEO_VECTOR_LEN) = np.array(raw_training_vectors).shape

print('Intersecting triangles and pruning')
training_vectors = []
target_vectors = []
for index, (a, b) in enumerate(triangle_sets):
    if loads(a).intersection(loads(b)).type == 'Polygon':
        intersection = loads(a).intersection(loads(b)).wkt
        target_vectors.append(GeoVectorizer.vectorize_wkt(intersection, max_points))
        training_vectors.append(raw_training_vectors[index])

training_vectors = np.array(training_vectors)
target_vectors = np.array(target_vectors)

inputs = Input(shape=(max_points, GEO_VECTOR_LEN))
model = LSTM(LATENT_SIZE, activation='relu', return_sequences=True)(inputs)
model = TimeDistributed(Dense(64, activation='relu'))(model)
model = Dense(GEO_VECTOR_LEN)(model)
model = Model(inputs, model)
model.compile(loss=geom_gaussian_loss, optimizer=OPTIMIZER)
model.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP, histogram_freq=1, write_graph=True)
my_callback = CustomCallback(GeoVectorizer.decypher)

history = model.fit(
    x=training_vectors,
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=[my_callback, tb_callback]
).history

print(history)
