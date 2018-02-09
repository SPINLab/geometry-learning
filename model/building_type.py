"""
This script executes the task of estimating the building type, based solely on the geometry for that building.
The data for this script can be generated by running the prep/get-data.sh and prep/preprocess-buildings.py scripts,
which will take about an hour or two.
"""
import os
from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Dense, Flatten
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

from topoml_util import geom_scaler
from topoml_util.slack_send import notify

SCRIPT_VERSION = '0.2.29'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
DATA_FOLDER = '../files/buildings/'
FILENAME_PREFIX = 'buildings-train'

# Hyperparameters
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1024))
TRAIN_VALIDATE_SPLIT = float(os.getenv('TRAIN_VALIDATE_SPLIT', 0.1))
REPEAT_DEEP_ARCH = int(os.getenv('REPEAT_DEEP_ARCH', 0))
LSTM_SIZE = int(os.getenv('LSTM_SIZE', 256))
DENSE_SIZE = int(os.getenv('DENSE_SIZE', 64))
EPOCHS = int(os.getenv('EPOCHS', 200))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 3e-4))
GEOM_SCALE = float(os.getenv('GEOM_SCALE', 0))  # Default 0, overridden when data is known
OPTIMIZER = Adam(lr=LEARNING_RATE)
PATIENCE = 40
RECURRENT_DROPOUT = 0.05


# Load training data
train_geoms = []
train_building_type = []

for file in os.listdir(DATA_FOLDER):
    if file.startswith(FILENAME_PREFIX) and file.endswith('.npz'):
        train_loaded = np.load(DATA_FOLDER + file)
        if len(train_geoms):
            train_geoms = np.append(train_geoms, train_loaded['geoms'], axis=0)
            train_building_type = np.append(train_building_type, train_loaded['building_type'], axis=0)
        else:
            train_geoms = train_loaded['geoms']
            train_building_type = train_loaded['building_type']

# Normalize
geom_scale = GEOM_SCALE or geom_scaler.scale(train_geoms)
train_geoms = geom_scaler.transform(train_geoms, geom_scale)

message = '''
running {0} with 
version: {1}                batch size: {2} 
train/validate split: {3}   repeat deep: {4} 
lstm size: {5}              dense size: {6} 
epochs: {7}                 learning rate: {8}
geometry scale: {:f}        recurrent dropout: {10}
patience {11}
'''.format(
    SIGNATURE,
    SCRIPT_VERSION, BATCH_SIZE,
    TRAIN_VALIDATE_SPLIT, REPEAT_DEEP_ARCH,
    LSTM_SIZE, DENSE_SIZE,
    EPOCHS, LEARNING_RATE,
    geom_scale, RECURRENT_DROPOUT,
    PATIENCE,
)
print(message)

# Map building types to one-hot vectors
train_targets = np.zeros((len(train_building_type), train_building_type.max() + 1))
for index, building_type in enumerate(train_building_type):
    train_targets[index, building_type] = 1

# Shape determination
geom_max_points, geom_vector_len = train_geoms.shape[1:]
output_seq_length = train_targets.shape[-1]

# Build model
inputs = Input(shape=(geom_max_points, geom_vector_len))
model = LSTM(LSTM_SIZE, return_sequences=True, recurrent_dropout=RECURRENT_DROPOUT)(inputs)

for layer in range(REPEAT_DEEP_ARCH):
    model = LSTM(LSTM_SIZE, return_sequences=True, recurrent_dropout=RECURRENT_DROPOUT)(model)
    # model = TimeDistributed(Dense(DENSE_SIZE, activation='relu'))(model)

model = Dense(DENSE_SIZE, activation='relu')(model)
model = Flatten()(model)
model = Dense(output_seq_length, activation='softmax')(model)

model = Model(inputs=inputs, outputs=model)
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=OPTIMIZER),
model.summary()

# Callbacks
callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + SIGNATURE, write_graph=False),
    EarlyStopping(patience=PATIENCE, min_delta=0.001)
]

history = model.fit(
    x=train_geoms,
    y=train_targets,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

# Run on unseen test data
TEST_DATA_FILE = '../files/buildings/buildings-test.npz'
test_loaded = np.load(TEST_DATA_FILE)
test_geoms = test_loaded['geoms']
test_building_types = test_loaded['building_type']

# Normalize
test_geoms = geom_scaler.transform(test_geoms, GEOM_SCALE)  # re-use variance from training

test_pred = [np.argmax(prediction) for prediction in model.predict(test_geoms)]
accuracy = accuracy_score(test_building_types, test_pred)
message = '''
test accuracy of {:f} with 
version: {1}                    batch size {2} 
train/validate split {3}        repeat deep arch {4} 
lstm size {5}                   dense size {6} 
epochs {7}                      learning rate {8}
geometry scale {:f}             recurrent dropout {10}
patience {11}
'''.format(
    accuracy,
    SCRIPT_VERSION,
    BATCH_SIZE,
    TRAIN_VALIDATE_SPLIT,
    REPEAT_DEEP_ARCH,
    LSTM_SIZE,
    DENSE_SIZE,
    len(history['val_loss']),
    LEARNING_RATE,
    geom_scale,
    RECURRENT_DROPOUT,
    PATIENCE,
)


notify(SIGNATURE, message)
print(SCRIPT_NAME, 'finished successfully')
