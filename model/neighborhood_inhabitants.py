import os
from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense, Flatten
from keras.optimizers import Adam

from topoml_util import geom_scaler
from topoml_util.slack_send import notify

SCRIPT_VERSION = '0.0.32'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
TRAINING_DATA_FILE = '../files/neighborhoods/neighborhoods_train.npz'

# Hyperparameters
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1024))
TRAIN_VALIDATE_SPLIT = float(os.getenv('TRAIN_VALIDATE_SPLIT', 0.1))
REPEAT_DEEP_ARCH = int(os.getenv('REPEAT_DEEP_ARCH', 0))
LSTM_SIZE = int(os.getenv('LSTM_SIZE', 256))
DENSE_SIZE = int(os.getenv('DENSE_SIZE', 64))
EPOCHS = int(os.getenv('EPOCHS', 200))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-4))
PATIENCE = 40
RECURRENT_DROPOUT = float(os.getenv('RECURRENT_DROPOUT', 0.1))
GEOM_SCALE = float(os.getenv('GEOM_SCALE', 0))  # If no default or 0: overridden when data is known
OPTIMIZER = Adam(lr=LEARNING_RATE)

train_loaded = np.load(TRAINING_DATA_FILE)
train_geoms = train_loaded['input_geoms']
train_above_or_below_median = train_loaded['above_or_below_median']

# Normalize
# means = geom_scaler.localized_mean(train_geoms)
# geom_scale = GEOM_SCALE or np.var(train_geoms[..., 0:2])
# train_geoms = geom_scaler.localized_normal(train_geoms, means, geom_scale)

geom_scale = GEOM_SCALE or geom_scaler.scale(train_geoms)
train_geoms = geom_scaler.transform(train_geoms, geom_scale)

message = '''
running {0} with 
version: {1}
batch size: {2} 
train/validate split: {3} 
repeat deep: {4} 
lstm size: {5} 
dense size: {6} 
epochs: {7} 
learning rate: {8}
geometry scale: {9}
recurrent dropout: {10}
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
)
print(message)

# Shape determination
geom_max_points, geom_vector_len = train_geoms.shape[1:]
output_seq_length = train_above_or_below_median.shape[-1]

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
    EarlyStopping(patience=PATIENCE, min_delta=0.001)
]

history = model.fit(
    x=train_geoms,
    y=train_above_or_below_median,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TRAIN_VALIDATE_SPLIT,
    callbacks=callbacks).history

# Run on unseen test data
TEST_DATA_FILE = '../files/neighborhoods/neighborhoods_test.npz'
test_loaded = np.load(TEST_DATA_FILE)
test_geoms = test_loaded['input_geoms']
test_above_or_below_median = test_loaded['above_or_below_median']

# Normalize
test_geoms = geom_scaler.transform(test_geoms, geom_scale)  # re-use variance from training
test_pred = model.predict(test_geoms)

correct = 0
for prediction, expected in zip(test_pred, test_above_or_below_median):
    if np.argmax(prediction) == np.argmax(expected):
        correct += 1

accuracy = correct / len(test_pred)
message = '''
test accuracy of {0} with '
version: {1} '
batch size {2} '
train/validate split {3} '
repeat deep arch {4} '
lstm size {5} '
dense size {6} '
epochs {7} '
learning rate {8}
geometry scale {9}
recurrent dropout {10}
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
)

notify(SIGNATURE, message)
print(SCRIPT_NAME, 'finished successfully')
