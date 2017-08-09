from keras.preprocessing.text import Tokenizer
import numpy as np


class Tokenize(Tokenizer):
    """Text tokenization wrapper around Keras text tokenization methods
    """

    def __init__(self, texts):
        super().__init__(num_words=None,
                         filters='\t\n',
                         lower=True,
                         split="",
                         char_level=True)
        self.fit_on_texts(texts)

    @staticmethod
    def truncate(max_len, untruncated_training_set, untruncated_target_set):
        """
        Method for truncating the training and target set to fit the maximum
            sequence length, batch and validation set size
        :param max_len: maximum length of characters per sequence/sentence
        :param untruncated_training_set: untruncated list of input sequences
        :param untruncated_target_set: untruncated list of target output sequences
        :return: training_set, target_set: a tuple of truncated training and target sets
        """
        training_set = []
        target_set = []

        # Restrict input to be of less or equal length as the maximum length.
        for index, record in enumerate(untruncated_training_set):
            if len(record) <= max_len:
                training_set.append(record)
                target_set.append(untruncated_target_set[index])

        return training_set, target_set

    @staticmethod
    def batch_truncate(batch_size, max_len, validation_split, untruncated_training_set, untruncated_target_set):
        """
        Method for truncating the training and target set to fit the maximum
            sequence length, batch and validation set size
        :param batch_size: size of the epoch batch size
        :param max_len: maximum length of characters per sequence/sentence
        :param validation_split: ratio of the training/validation split
        :param untruncated_training_set: untruncated list of input sequences
        :param untruncated_target_set: untruncated list of target output sequences
        :return: training_set, target_set: a tuple of truncated training and target sets
        """
        training_set = []
        target_set = []

        # Restrict input to be of less or equal length as the maximum length.
        for index, record in enumerate(untruncated_training_set):
            if len(record) <= max_len:
                training_set.append(record)
                target_set.append(untruncated_target_set[index])

        # Truncate the array to the batch size, accounting for the validation set
        # The validation sample size must be a multiple of the batch size
        # Say the truncated length is 27,000 and the split ratio is 0.1, the validation sample size is 2700
        validation_size = int(len(training_set) * validation_split)
        # We need to get it down to 2000
        validation_size = validation_size - validation_size % batch_size
        # The truncated length must be a multiple of the validation sample size
        truncated_size = len(training_set) - len(training_set) % int(validation_size / validation_split)
        training_set = training_set[0:truncated_size]
        target_set = target_set[0:truncated_size]
        return training_set, target_set

    @staticmethod
    def max_sample(predictions):
        # helper function to sample an index from a probability array
        return np.argmax(predictions)

    def char_level_tokenize(self, texts):
        sequences = self.texts_to_sequences(texts)
        return sequences

    def decypher(self, sequences):
        """
        Decyphers a encoded 3D array of one-hot vectors back to a 2D array of sentences
        :param sequences:
        :return:
        """
        # sampled = [Tokenize.max_sample(token) for token in prediction]
        # sequence.append(sampled)
        inv_cipher = {v: k for k, v in self.word_index.items()}
        decyphered = []
        for sequence in sequences:
            decyphered_sequence = []
            for num in sequence:
                if num in inv_cipher:
                    decyphered_sequence.append(inv_cipher[num])
                else:
                    decyphered_sequence.append(' ')
            decyphered.append(''.join([char for char in decyphered_sequence]))
        return decyphered

    def one_hot(self, input_sequences, maxlen):
        # The third dimension of the matrix is equal to the length of the word index plus one:
        # There is no '0' index in the word index.
        x = np.zeros((len(input_sequences), maxlen, len(self.word_index) + 1), dtype=np.bool)
        for i, sentence in enumerate(input_sequences):
            for t, char in enumerate(sentence):
                x[i, t, self.word_index[char]] = True
        return x

