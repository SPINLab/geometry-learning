"""
This script executes the task of estimating the number of inhabitants of a neighborhood to be under or over the
median of all neighborhoods in the Netherlands, based solely on the geometry for that neighborhood.
The data for this script can be found at http://hdl.handle.net/10411/GYPPBR.
"""

import os
import socket
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import time
from urllib.request import urlretrieve

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from topoml_util import geom_scaler
from topoml_util.slack_send import notify

SCRIPT_VERSION = '2.0.5'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + SCRIPT_VERSION + ' ' + TIMESTAMP
DATA_FOLDER = '../files/neighborhoods/'
TRAIN_DATA_FILE = 'neighborhoods_train_v7.npz'
TEST_DATA_FILE = 'neighborhoods_test_v7.npz'
TRAIN_DATA_URL = 'https://dataverse.nl/api/access/datafile/11378'
TEST_DATA_URL = 'https://dataverse.nl/api/access/datafile/11379'
SCRIPT_START = time()

# Hyperparameters
hp = {
    'BATCH_SIZE': int(os.getenv('BATCH_SIZE', 32)),
    'TRAIN_VALIDATE_SPLIT': float(os.getenv('TRAIN_VALIDATE_SPLIT', 0.1)),
    'REPEAT_DEEP_ARCH': int(os.getenv('REPEAT_DEEP_ARCH', 0)),
    'DENSE_SIZE': int(os.getenv('DENSE_SIZE', 32)),
    'EPOCHS': int(os.getenv('EPOCHS', 200)),
    'LEARNING_RATE': float(os.getenv('LEARNING_RATE', 1e-3)),
    'DROPOUT': float(os.getenv('DROPOUT', 0.0)),
    'GEOM_SCALE': float(os.getenv("GEOM_SCALE", 0)),  # If no default or 0: overridden when data is known
}
OPTIMIZER = Adam(lr=hp['LEARNING_RATE'])

# Load training data
path = Path(DATA_FOLDER + TRAIN_DATA_FILE)
if not path.exists():
    print("Retrieving training data from web...")
    urlretrieve(TRAIN_DATA_URL, DATA_FOLDER + TRAIN_DATA_FILE)

train_loaded = np.load(DATA_FOLDER + TRAIN_DATA_FILE)
train_geoms = train_loaded['fixed_size_geoms']
train_labels = train_loaded['above_or_below_median']

# Determine final test mode or standard
if len(sys.argv) > 1 and sys.argv[1] in ['-t', '--test']:
    print('Training in final test mode')
    path = Path(DATA_FOLDER + TEST_DATA_FILE)
    if not path.exists():
        print("Retrieving test data from web...")
        urlretrieve(TEST_DATA_URL, DATA_FOLDER + TEST_DATA_FILE)

    test_loaded = np.load(DATA_FOLDER + TEST_DATA_FILE)
    test_geoms = test_loaded['fixed_size_geoms']
    test_labels = test_loaded['above_or_below_median']
else:
    print('Training in standard training mode')
    # Split the training data in random seen/unseen sets
    train_geoms, test_geoms, train_labels, test_labels = train_test_split(train_geoms, train_labels, test_size=0.1)

# Normalize
geom_scale = hp['GEOM_SCALE'] or geom_scaler.scale(train_geoms)
train_geoms = geom_scaler.transform(train_geoms, geom_scale)
test_geoms = geom_scaler.transform(test_geoms, geom_scale)  # re-use variance from training

# Map types to one-hot vectors
# noinspection PyUnresolvedReferences
train_targets = np.zeros((len(train_labels), train_labels.max() + 1))
for index, train_label in enumerate(train_labels):
    train_targets[index, train_label] = 1

# Shape determination
geom_vector_len = train_geoms[0].shape[1]

# Build model
inputs = Input(shape=(None, geom_vector_len))
model = Conv1D(32, (5,), activation='relu')(inputs)
model = MaxPooling1D(3, padding='SAME')(model)
model = Conv1D(64, (5,), activation='relu')(model)
model = GlobalAveragePooling1D()(model)
model = Dense(hp['DENSE_SIZE'], activation='relu')(model)
model = Dropout(hp['DROPOUT'])(model)
model = Dense(2, activation='softmax')(model)

model = Model(inputs=inputs, outputs=model)
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=OPTIMIZER),
model.summary()

# Callbacks
callbacks = [TensorBoard(log_dir='./tensorboard_log/' + SIGNATURE, write_graph=False)]

history = model.fit(
    x=train_geoms,
    y=train_targets,
    epochs=hp['EPOCHS'],
    batch_size=hp['BATCH_SIZE'],
    validation_split=hp['TRAIN_VALIDATE_SPLIT'],
    callbacks=callbacks).history

# Run on unseen test data
test_pred = [np.argmax(prediction) for prediction in model.predict(test_geoms)]
accuracy = accuracy_score(test_labels, test_pred)

runtime = time() - SCRIPT_START
message = 'on {} completed with accuracy of \n{:f} \nin {} in {} epochs\n'.format(
    socket.gethostname(), accuracy, timedelta(seconds=runtime), len(history['val_loss']))

for key, value in sorted(hp.items()):
    message += '{}: {}\t'.format(key, value)

notify(SIGNATURE, message)
print(SCRIPT_NAME, 'finished successfully with', message)
