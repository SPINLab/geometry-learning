import pandas
import tensorflow as tf
from keras import Input
from keras.activations import tanh
from keras.engine import Model
from keras.layers import LSTM, Dense
from topoml_util.CustomCallback import CustomCallback
from topoml_util.Tokenizer import Tokenize

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

TOPOLOGY_TRAINING_CSV = '../files/topology-training.csv'
MAX_SEQUENCE_LEN = 220
TRAIN_VALIDATE_SPLIT = 0.1
EPOCHS = 30


def main():
    print('Reading data...')
    csv_matrix = pandas.read_csv(TOPOLOGY_TRAINING_CSV).as_matrix()
    print(len(csv_matrix), 'data points in training set')

    training_set = [record[0] + ' ' + record[1] for record in csv_matrix]
    target_set = [record[2] for record in csv_matrix]
    (training_set, target_set) = Tokenize.truncate(MAX_SEQUENCE_LEN, training_set, target_set)
    print(len(target_set), 'max length data points in training set')

    print('Tokenizing string sequences...')
    tokenizer = Tokenize(training_set + target_set)  # Initialize with the full set
    input_one_hot = tokenizer.one_hot(training_set, MAX_SEQUENCE_LEN)
    target_one_hot = tokenizer.one_hot(target_set, MAX_SEQUENCE_LEN)

    print('Building model...')
    # Adapted from example https://blog.keras.io/building-autoencoders-in-keras.html
    word_index_size = len(tokenizer.word_index) + 1
    inputs = Input(shape=(MAX_SEQUENCE_LEN, word_index_size))
    encoded = LSTM(word_index_size, name='Encoding_LSTM', return_sequences=True)(inputs)
    encoded = LSTM(word_index_size, name='Hidden_LSTM', return_sequences=True)(encoded)
    encoded = LSTM(word_index_size, name='Hidden_LSTM2', return_sequences=True)(encoded)
    # decoded = LSTM(word_index_size, return_sequences=True, name='Decoding_LSTM')(inputs)
    encoder = Model(inputs=inputs, outputs=encoded)

    encoder.summary()
    encoder.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # encoder = Model(inputs=inputs, outputs=encoded)

    # tb_callback = TensorBoard(log_dir='./tensorboard_log', histogram_freq=1, write_graph=True, write_images=True)
    my_callback = CustomCallback(tokenizer.decypher)

    encoder.fit(x=input_one_hot,
                y=target_one_hot,
                epochs=EPOCHS,
                batch_size=1000,
                validation_split=TRAIN_VALIDATE_SPLIT,
                callbacks=[my_callback])

    print('Done!')


if __name__ == '__main__':
    main()
