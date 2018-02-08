import os
from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense, Flatten
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

from topoml_util import geom_scaler
from topoml_util.slack_send import notify

SCRIPT_VERSION = '0.0.4'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
TRAINING_DATA_FILE = '../files/archaeology/archaeo_features_train.npz'

# Hyperparameters
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1024))
TRAIN_VALIDATE_SPLIT = float(os.getenv('TRAIN_VALIDATE_SPLIT', 0.1))
REPEAT_DEEP_ARCH = int(os.getenv('REPEAT_DEEP_ARCH', 0))
LSTM_SIZE = int(os.getenv('LSTM_SIZE', 256))
DENSE_SIZE = int(os.getenv('DENSE_SIZE', 64))
EPOCHS = int(os.getenv('EPOCHS', 200))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-4))
PATIENCE = int(os.getenv('PATIENCE', 16))
RECURRENT_DROPOUT = float(os.getenv('RECURRENT_DROPOUT', 0.05))
GEOM_SCALE = float(os.getenv('GEOM_SCALE', 0))  # If no default or 0: overridden when data is known
OPTIMIZER = Adam(lr=LEARNING_RATE)

train_loaded = np.load(TRAINING_DATA_FILE)
train_geoms = train_loaded['geoms']
train_feature_type = train_loaded['feature_type']

# Normalize
geom_scale = GEOM_SCALE or geom_scaler.scale(train_geoms)
train_geoms = geom_scaler.transform(train_geoms, geom_scale)

# Map types to one-hot vectors
train_targets = np.zeros((len(train_feature_type), train_feature_type.max() + 1))
for index, feature_type in enumerate(train_feature_type):
    train_targets[index, feature_type] = 1

message = '''
running {0} with 
version: {1}                batch size: {2} 
train/validate split: {3}   repeat deep: {4} 
lstm size: {5}              dense size: {6} 
epochs: {7}                 learning rate: {8}
geometry scale: {9}         recurrent dropout: {10}
patience {11}
'''.format(
    SIGNATURE,
    SCRIPT_VERSION,
    BATCH_SIZE,
    TRAIN_VALIDATE_SPLIT,
    REPEAT_DEEP_ARCH,
    LSTM_SIZE,
    DENSE_SIZE,
    EPOCHS,
    LEARNING_RATE,
    geom_scale,
    RECURRENT_DROPOUT,
    PATIENCE,
)
print(message)

# Shape determination
geom_max_points, geom_vector_len = train_geoms.shape[1:]
output_seq_length = train_targets.shape[-1]

# Build model
inputs = Input(shape=(geom_max_points, geom_vector_len))
model = LSTM(LSTM_SIZE, return_sequences=True, recurrent_dropout=RECURRENT_DROPOUT)(inputs)
model = TimeDistributed(Dense(DENSE_SIZE, activation='relu'))(model)

for layer in range(REPEAT_DEEP_ARCH):
    model = LSTM(LSTM_SIZE, return_sequences=True, recurrent_dropout=RECURRENT_DROPOUT)(model)

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
    EarlyStopping(patience=PATIENCE, min_delta=0.001),
]

history = model.fit(
    x=train_geoms,
    y=train_targets,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

# Run on unseen test data
TEST_DATA_FILE = '../files/archaeology/archaeo_features_test.npz'
test_loaded = np.load(TEST_DATA_FILE)
test_geoms = test_loaded['geoms']
test_feature_type = test_loaded['feature_type']
test_geoms = geom_scaler.transform(test_geoms, geom_scale)  # re-use variance from training

test_pred = model.predict(test_geoms)
accuracy = accuracy_score(test_feature_type, test_pred)
message = '''
test accuracy of {0} with 
version: {1}                batch size {2} 
train/validate split {3}    repeat deep arch {4} 
lstm size {5}               dense size {6} 
epochs {7}                  learning rate {8}
geometry scale {9}          recurrent dropout {10}
patience {11}
'''.format(
    str(accuracy),
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
