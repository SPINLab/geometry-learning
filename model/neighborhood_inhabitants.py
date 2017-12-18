import os
from datetime import datetime

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, TimeDistributed, Dense, Flatten
from keras.optimizers import Adam

from topoml_util.geom_scaler import localized_mean, localized_normal
from topoml_util.slack_send import notify

SCRIPT_VERSION = '0.0.4'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
TRAINING_DATA_FILE = '../files/neighborhoods/neighborhoods_train.npz'

# Hyperparameters
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 64))
TRAIN_VALIDATE_SPLIT = float(os.getenv('TRAIN_VALIDATE_SPLIT', 0.1))
REPEAT_DEEP_ARCH = int(os.getenv('REPEAT_DEEP_ARCH', 0))
LSTM_SIZE = int(os.getenv('LSTM_SIZE', 128))
DENSE_SIZE = int(os.getenv('DENSE_SIZE', 64))
EPOCHS = int(os.getenv('EPOCHS', 400))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-4))

OPTIMIZER = Adam(lr=LEARNING_RATE)

train_loaded = np.load(TRAINING_DATA_FILE)
train_geoms = train_loaded['input_geoms']
train_above_or_below_median = train_loaded['above_or_below_median']

# Normalize
means = localized_mean(train_geoms)
variance = np.var(train_geoms[..., 0:2])
train_geoms = localized_normal(train_geoms, means, variance)

# Shape determination
geom_max_points, geom_vector_len = train_geoms.shape[1:]
output_seq_length = train_above_or_below_median.shape[-1]

# Build model
inputs = Input(shape=(geom_max_points, geom_vector_len))
model = LSTM(geom_max_points, activation='relu', return_sequences=True)(inputs)

for layer in range(REPEAT_DEEP_ARCH):
    model = LSTM(LSTM_SIZE, activation='relu', return_sequences=True)(model)

model = TimeDistributed(Dense(DENSE_SIZE, activation='relu'))(model)
model = Flatten()(model)
model = Dense(output_seq_length)(model)

model = Model(inputs=inputs, outputs=model)
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=OPTIMIZER),
model.summary()

# Callbacks
callbacks = [
    TensorBoard(log_dir='./tensorboard_log/' + SIGNATURE, write_graph=False),
    EarlyStopping(patience=40, min_delta=0.01)
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
means = localized_mean(test_geoms)
test_geoms = localized_normal(test_geoms, means, variance)  # re-use variance from training
test_pred = model.predict(test_geoms)

correct = 0
for prediction, expected in zip(test_pred, test_above_or_below_median):
    if all([pred == exp for pred, exp in zip(prediction, expected)]):
        correct += 1

accuracy = correct / len(test_pred)
message = 'test accuracy of {0} with ' \
          'batch size {1} ' \
          'train/validate split {2} ' \
          'repeat deep arch {3} ' \
          'lstm size {4} ' \
          'epochs {5} ' \
          'learning rate {6}'\
    .format(
        str(accuracy),
        BATCH_SIZE,
        TRAIN_VALIDATE_SPLIT,
        REPEAT_DEEP_ARCH,
        LSTM_SIZE,
        DENSE_SIZE,
        EPOCHS,
        LEARNING_RATE)

notify(SIGNATURE, message)
print(SCRIPT_NAME, 'finished successfully')
