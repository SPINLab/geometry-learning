import os
from datetime import datetime
import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, Dense, TimeDistributed
from keras.optimizers import Adam
from shutil import copyfile

from shapely.geometry import Polygon

from topoml_util.GeoVectorizer import GeoVectorizer
from topoml_util.LoggerCallback import EpochLogger
from topoml_util.wkt2pyplot import wkt2pyplot

SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
DATA_FILE = '../files/triangles.npz'
BATCH_SIZE = 2048
TRAIN_VALIDATE_SPLIT = 0.1
LATENT_SIZE = 64
EPOCHS = 400
OPTIMIZER = Adam(lr=1e-2, decay=1e-4)

# Archive the configuration
copyfile(__file__, 'configs/' + TIMESTAMP + ' ' + SCRIPT_NAME)

loaded = np.load(DATA_FILE)
training_vectors = loaded['point_sequence']
(set_size, GEO_VECTOR_LEN) = training_vectors.shape
training_vectors = np.reshape(training_vectors, (set_size, 1, GEO_VECTOR_LEN))
target_vectors = loaded['intersection_surface']

inputs = Input(shape=(1, GEO_VECTOR_LEN))
model = LSTM(LATENT_SIZE, activation='relu', return_sequences=True)(inputs)
model = Dense(32, activation='relu')(model)
model = LSTM(LATENT_SIZE, activation='relu')(model)
model = Dense(32, activation='relu')(model)
model = Dense(1)(model)
model = Model(inputs, model)
model.compile(loss='mse', optimizer=OPTIMIZER)
model.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP + ' ' + SCRIPT_NAME,
                          histogram_freq=1, write_graph=True)
epoch_callback = EpochLogger(
    input_func=GeoVectorizer.decypher,
    target_func=lambda x: str(x),
    predict_func=lambda x: str(x),
    aggregate_func=None,
    stdout=True
)

model.fit(
    x=training_vectors,
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=[epoch_callback, tb_callback]
)

plot_sample = training_vectors[-10000:]
prediction = model.predict(plot_sample)
intersecting_error = np.abs(prediction[:, 0] - target_vectors[-10000:])

triangle_vectors = plot_sample.reshape(10000, 6, 2)
triangle_sets = np.array([[Polygon(point_set[0:3]).wkt, Polygon(point_set[3:]).wkt]
                          for point_set in triangle_vectors])
zipped = zip(triangle_sets, target_vectors[-10000:], prediction[:, 0])
sorted_results = sorted(zipped, key=lambda record: abs(record[2] - record[1]))

print('Intersection surface area mean:', np.mean(target_vectors))
print('Intersecting error mean:', np.mean(intersecting_error))

plot_samples = 50
print('Saving top and bottom', plot_samples, 'results as plots, this will take a few minutes...')
# print('Worst', plot_samples, 'results: ', sorted_results[-plot_samples:])
for result in sorted_results[-plot_samples:]:
    timestamp = str(datetime.now()).replace(':', '.')
    plot, fig, ax = wkt2pyplot(result[0])
    plot.text(0.01, 0.06, 'target: ' + str(result[1]), transform=ax.transAxes)
    plot.text(0.01, 0.01, 'prediction: ' + str(result[2]), transform=ax.transAxes)
    plot.savefig('./plots/bad_' + timestamp + '.png')
    plot.close()

# print('Best', plot_samples, 'results:', sorted_results[0:plot_samples])
for result in sorted_results[0:plot_samples]:
    timestamp = str(datetime.now()).replace(':', '.')
    plot, fig, ax = wkt2pyplot(result[0])
    plot.text(0.01, 0.06, 'target: ' + str(result[1]), transform=ax.transAxes)
    plot.text(0.01, 0.01, 'prediction: ' + str(result[2]), transform=ax.transAxes)
    plot.savefig('./plots/good_' + timestamp + '.png')
    plot.close()

print('Done!')
