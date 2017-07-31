import numpy as np
import pandas
from keras import Input
from keras.backend import floatx
from keras.engine import Model
from keras.layers import LSTM, RepeatVector
from topoml_util.util import Tokenize

TOPOLOGY_TRAINING_CSV = '../files/topology-training.csv'

def main():
    print('Reading data...')
    training_data = pandas.read_csv(TOPOLOGY_TRAINING_CSV)
    raw_training_set = training_data['brt_wkt'] + ' ' + training_data['osm_wkt']
    raw_target_set = training_data['intersection_wkt']
    print(len(raw_training_set), 'data points in training set')

    training_set = []
    target_set = []

    max_len = 500
    batch_size = 100

    # Restrict input to be of less or equal length as the maximum length.
    for index, record in enumerate(raw_training_set):
        if len(record) <= max_len:
            training_set.append(record)
            target_set.append(raw_target_set[index])

    # Truncate the array to the batch size
    truncated_length = len(training_set) - len(training_set) % batch_size
    training_set = training_set[0:truncated_length]
    target_set = target_set[0:truncated_length]

    print(len(target_set), 'max length filtered data points in training set')

    tokenizer = Tokenize(''.join(training_set + target_set))
    input_one_hot = tokenizer.one_hot(training_set, max_len)
    target_one_hot = tokenizer.one_hot(target_set, max_len)

    print('Building model...')
    # Adapted from example https://blog.keras.io/building-autoencoders-in-keras.html
    latent_dim = 128
    inputs = Input(batch_shape=(batch_size, max_len, len(tokenizer.word_index) + 1))
    encoded = LSTM(latent_dim, name='Encoding_LSTM')(inputs)
    decoded = LSTM(len(tokenizer.word_index) + 1, return_sequences=True, name='Decoding_LSTM')(inputs)
    sequence_autoencoder = Model(inputs=inputs, outputs=decoded)
    encoder = Model(inputs=inputs, outputs=encoded)
    sequence_autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop')
    sequence_autoencoder.fit(x=input_one_hot, y=target_one_hot, epochs=20, batch_size=batch_size)
    print('Done!')


if __name__ == '__main__':
    main()
