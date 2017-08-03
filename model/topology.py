import numpy as np
import pandas
from keras import Input
from keras.backend import floatx
from keras.callbacks import TensorBoard, RemoteMonitor
from keras.engine import Model
from keras.layers import LSTM, RepeatVector
from topoml_util.Tokenizer import Tokenize
from topoml_util.CustomCallback import CustomCallback

TOPOLOGY_TRAINING_CSV = '../files/topology-training.csv'


def main():
    print('Reading data...')
    training_data = pandas.read_csv(TOPOLOGY_TRAINING_CSV)
    raw_training_set = training_data['brt_wkt'] + ' ' + training_data['osm_wkt']
    raw_target_set = training_data['intersection_wkt']
    print(len(raw_training_set), 'data points in training set')

    max_len = 250
    train_validate_split = 0.1

    target_set, training_set = Tokenize.truncate(max_len,
                                                 raw_training_set,
                                                 raw_target_set)

    print(len(target_set), 'max length data points in training set')

    print('Tokenizing string sequences...')
    tokenizer = Tokenize(''.join(training_set + target_set))
    input_one_hot = tokenizer.one_hot(training_set, max_len)
    target_one_hot = tokenizer.one_hot(target_set, max_len)

    print('Building model...')
    # Adapted from example https://blog.keras.io/building-autoencoders-in-keras.html
    # latent_dim = 128
    inputs = Input(shape=(max_len, len(tokenizer.word_index) + 1))
    # encoded = LSTM(latent_dim, name='Encoding_LSTM')(inputs)
    decoded = LSTM(len(tokenizer.word_index) + 1, return_sequences=True, name='Decoding_LSTM')(inputs)
    sequence_autoencoder = Model(inputs=inputs, outputs=decoded)
    sequence_autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # encoder = Model(inputs=inputs, outputs=encoded)

    # tb_callback = TensorBoard(log_dir='./tensorboard_log', histogram_freq=1, write_graph=True, write_images=True)
    my_callback = CustomCallback(tokenizer.decypher)

    sequence_autoencoder.fit(x=input_one_hot,
                             y=target_one_hot,
                             epochs=10,
                             batch_size=1000,
                             validation_split=train_validate_split,
                             callbacks=[my_callback])
    print('Done!')


if __name__ == '__main__':
    main()
