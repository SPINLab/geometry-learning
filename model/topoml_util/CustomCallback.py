from keras.callbacks import Callback
from .Tokenizer import Tokenize


class CustomCallback(Callback):
    def __init__(self, decypher):
        super().__init__()
        self.decypher = decypher

    def on_epoch_end(self, epoch, logs=None):
        validation_sample = self.validation_data[0][0:1]
        predictions = self.model.predict(validation_sample)
        sequence = []
        for prediction in predictions:
            sampled = [Tokenize.max_sample(token) for token in prediction]
            sequence.append(sampled)
        print('\nPrediction: %s' % self.decypher(sequence))
