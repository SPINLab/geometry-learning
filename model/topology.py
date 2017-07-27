import numpy as np
import pandas
from keras import Input
from keras.engine import Model
from keras.layers import LSTM, RepeatVector

TOPOLOGY_TRAINING_CSV = '../files/topology-training.csv'


def main():
    print('Reading data...')
    training_data = pandas.read_csv(TOPOLOGY_TRAINING_CSV)
    raw_training_set = training_data['brt_wkt'] + '; ' + training_data['osm_wkt']
    raw_target_set = training_data['intersection_wkt']
    print(len(raw_training_set), 'data points in training set')

    training_set = []
    target_set = []

    max_len = 1000
    # Restrict input to be of less or equal length as the maximum length.
    for index, record in enumerate(raw_training_set):
        if len(record) <= max_len:
            training_set.append(record)
            target_set.append(raw_target_set[index])

    print(len(target_set), 'max length filtered data points in training set')

    for geometry in training_set:
        if len(geometry) > max_len:
            max_len = len(geometry)

    print('Building model...')
    latent_dim = 100
    inputs = Input(shape=(max_len, None), dtype=str)
    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(len(training_set))(encoded)
    decoded = LSTM(max_len, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    sequence_autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop')
    sequence_autoencoder.fit(x=np.array([training_set]), y=np.array([target_set]), epochs=20)

    print('Done!')


if __name__ == '__main__':
    main()
