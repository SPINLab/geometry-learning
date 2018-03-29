import os
import socket
import sys
from datetime import datetime, timedelta
from time import time

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense, Flatten, Bidirectional
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from topoml_util import geom_scaler
from topoml_util.slack_send import notify

SCRIPT_VERSION = '1.0.3'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + SCRIPT_VERSION + ' ' + TIMESTAMP
TRAINING_DATA_FILE = '../files/archaeology/archaeo_features_train.npz'
SCRIPT_START = time()

# Hyperparameters
hp = {
    'BATCH_SIZE': int(os.getenv('BATCH_SIZE', 512)),
    'TRAIN_VALIDATE_SPLIT': float(os.getenv('TRAIN_VALIDATE_SPLIT', 0.1)),
    'REPEAT_DEEP_ARCH': int(os.getenv('REPEAT_DEEP_ARCH', 0)),
    'LSTM_SIZE': int(os.getenv('LSTM_SIZE', 32)),
    'DENSE_SIZE': int(os.getenv('DENSE_SIZE', 32)),
    'EPOCHS': int(os.getenv('EPOCHS', 200)),
    'LEARNING_RATE': float(os.getenv('LEARNING_RATE', 8e-3)),
    'PATIENCE': int(os.getenv('PATIENCE', 16)),
    'RECURRENT_DROPOUT': float(os.getenv('RECURRENT_DROPOUT', 0.0)),
    'GEOM_SCALE': float(os.getenv("GEOM_SCALE", 0)),  # If no default or 0: overridden when data is known
    'EARLY_STOPPING': bool(os.getenv('EARLY_STOPPING', False)),
}
OPTIMIZER = Adam(lr=hp['LEARNING_RATE'], clipnorm=1.)

train_loaded = np.load(TRAINING_DATA_FILE)
train_geoms = train_loaded['geoms']
train_labels = train_loaded['feature_type']

# Determine final test mode or standard
if len(sys.argv) > 1 and sys.argv[1] in ['-t', '--test']:
    print('Training in final test mode')
    TEST_DATA_FILE = '../files/archaeology/archaeo_features_test.npz'
    test_loaded = np.load(TEST_DATA_FILE)
    test_geoms = test_loaded['geoms']
    test_labels = test_loaded['feature_type']
else:
    print('Training in standard training mode')
    # Split the training data in random seen/unseen sets
    train_geoms, test_geoms, train_labels, test_labels = train_test_split(train_geoms, train_labels, test_size=0.1)

# Normalize
geom_scale = hp['GEOM_SCALE'] or geom_scaler.scale(train_geoms)
train_geoms = geom_scaler.transform(train_geoms, geom_scale)
test_geoms = geom_scaler.transform(test_geoms, geom_scale)  # re-use variance from training

# Map types to one-hot vectors
train_targets = np.zeros((len(train_labels), train_labels.max() + 1))
for index, feature_type in enumerate(train_labels):
    train_targets[index, feature_type] = 1

# Shape determination
geom_max_points, geom_vector_len = train_geoms.shape[1:]
output_size = train_targets.shape[-1]

# Build model
inputs = Input(shape=(geom_max_points, geom_vector_len))
model = Bidirectional(LSTM(hp['LSTM_SIZE'],
                           return_sequences=(hp['REPEAT_DEEP_ARCH'] > 0),
                           recurrent_dropout=hp['RECURRENT_DROPOUT']))(inputs)

for layer in range(hp['REPEAT_DEEP_ARCH']):
    is_last_layer = (layer + 1 == hp['REPEAT_DEEP_ARCH'])
    model = Bidirectional(LSTM(hp['LSTM_SIZE'],
                               return_sequences=(not is_last_layer),
                               recurrent_dropout=hp['RECURRENT_DROPOUT']))(model)

model = Dense(output_size, activation='softmax')(model)

model = Model(inputs=inputs, outputs=model)
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=OPTIMIZER),
model.summary()

# Callbacks
callbacks = [TensorBoard(log_dir='./tensorboard_log/' + SIGNATURE, write_graph=False)]
if hp['EARLY_STOPPING']:
    callbacks.append(EarlyStopping(patience=hp['PATIENCE'], min_delta=0.001))

history = model.fit(
    x=train_geoms,
    y=train_targets,
    epochs=hp['EPOCHS'],
    batch_size=hp['BATCH_SIZE'],
    validation_split=hp['TRAIN_VALIDATE_SPLIT'],
    callbacks=callbacks).history

# Run on unseen test data
test_pred = [np.argmax(classes) for classes in model.predict(test_geoms)]
accuracy = accuracy_score(test_labels, test_pred)

runtime = time() - SCRIPT_START
message = 'on {} completed with accuracy of \n{:f} \nin {} in {} epochs\n'.format(
    socket.gethostname(), accuracy, timedelta(seconds=runtime), len(history['val_loss']))

for key, value in sorted(hp.items()):
    message += '{}: {}\t'.format(key, value)

notify(SIGNATURE, message)
print(SCRIPT_NAME, 'finished successfully with', message)
