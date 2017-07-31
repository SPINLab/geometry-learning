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

    def char_level_tokenize(self, texts):
        sequences = self.texts_to_sequences(texts)
        return sequences

    def detokenize(self, sequences):
        inv_cipher = {v: k for k, v in self.word_index.items()}
        deciphered = []
        for sequence in sequences:
            deciphered_sequence = ''.join([inv_cipher[c] for c in sequence])
            deciphered.append(deciphered_sequence)
        return deciphered

    def one_hot(self, input_sequences, maxlen):
        # The third dimension of the matrix is equal to the length of the word index plus one:
        # There is no 0 index in the word index.
        X = np.zeros((len(input_sequences), maxlen, len(self.word_index) + 1), dtype=np.bool)
        for i, sentence in enumerate(input_sequences):
            for t, char in enumerate(sentence):
                X[i, t, self.word_index[char]] = True
        return X
