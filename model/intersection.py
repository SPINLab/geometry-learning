import os
import numpy as np
from datetime import datetime
from shutil import copyfile
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

from topoml_util.GeoVectorizer import GeoVectorizer, GEOM_TYPE_LEN, RENDER_LEN
from topoml_util.GaussianMixtureLoss import GaussianMixtureLoss
from topoml_util.PyplotLogger import DecypherAll
from topoml_util.geom_scaler import localized_normal, localized_mean
from topoml_util.wkt2pyplot import wkt2pyplot

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
PLOT_DIR = './plots/' + TIMESTAMP + ' ' + SCRIPT_NAME
DATA_FILE = '../files/geodata_vectorized.npz'
BATCH_SIZE = 2048
GAUSSIAN_MIXTURE_COMPONENTS = 1
TRAIN_VALIDATE_SPLIT = 0.1
REPEAT_DEEP_ARCH = 2
LATENT_SIZE = 128
EPOCHS = 200
OPTIMIZER = Adam(lr=1e-3)

# Archive the configuration
copyfile(__file__, 'configs/' + TIMESTAMP + ' ' + SCRIPT_NAME)

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
# Expand the target vectors to gaussian mixture model size compatible with the prediction format
component_1 = target_vectors[:, :, 0:5]  # The first gaussian mixture model component for each point
component_1 = np.append(component_1, np.zeros((data_points, max_points, 1)), axis=2)  # Add pi feature
target_vectors = np.append(
    np.reshape(
        np.repeat(component_1, GAUSSIAN_MIXTURE_COMPONENTS, axis=1),
        (data_points, max_points, 6 * GAUSSIAN_MIXTURE_COMPONENTS)),
    target_vectors[:, :, 5:], axis=2)

output_size = GAUSSIAN_MIXTURE_COMPONENTS * 6 + GEOM_TYPE_LEN + RENDER_LEN

inputs = Input(shape=(max_points, INPUT_VECTOR_LEN))
model = Dense(64, activation='relu')(inputs)

for layer in range(REPEAT_DEEP_ARCH):
    model = LSTM(LATENT_SIZE, activation='relu', return_sequences=True)(model)
    model = Dense(64, activation='relu')(model)

model = Dense(output_size)(model)
model = Model(inputs, model)
model.compile(
    loss=GaussianMixtureLoss(GAUSSIAN_MIXTURE_COMPONENTS, max_points).geom_gaussian_mixture_loss,
    optimizer=OPTIMIZER)
model.summary()

tb_callback = TensorBoard(log_dir='./tensorboard_log/' + TIMESTAMP + ' ' + SCRIPT_NAME,
                          histogram_freq=1, write_graph=True)
decypher = DecypherAll(gmm_size=GAUSSIAN_MIXTURE_COMPONENTS, plot_dir=PLOT_DIR)

model.fit(
    x=training_vectors,
    y=target_vectors,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=[decypher, tb_callback])

prediction = model.predict(training_vectors[-10000:])
intersecting_error = np.abs(prediction[:, 0] - target_vectors[0:1000, 0])

decyphered = [GeoVectorizer.decypher(vector).split('\n') for vector in training_vectors[0:1000]]
zipped = zip(decyphered, target_vectors[0:1000, 0], prediction[:, 0])
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
    plot.savefig('./plot_images/bad_' + timestamp + '.png')
    plot.close()

# print('Best', plot_samples, 'results:', sorted_results[0:plot_samples])
for result in sorted_results[0:plot_samples]:
    timestamp = str(datetime.now()).replace(':', '.')
    plot, fig, ax = wkt2pyplot(result[0])
    plot.text(0.01, 0.06, 'target: ' + str(result[1]), transform=ax.transAxes)
    plot.text(0.01, 0.01, 'prediction: ' + str(result[2]), transform=ax.transAxes)
    plot.savefig('./plot_images/good_' + timestamp + '.png')
    plot.close()

print('Done!')

