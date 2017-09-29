from keras.callbacks import Callback
import random
from datetime import datetime
import numpy as np


class DecypherAll(Callback):
    def __init__(self, decypher):
        super().__init__()
        self.decypher = decypher

    def on_epoch_end(self, epoch, logs=None):
        random.seed(datetime.now())
        sample_indexes = random.sample(range(len(self.validation_data[0])), 3)
        input_samples = [self.validation_data[0][sample] for sample in sample_indexes]
        target_samples = [self.validation_data[1][sample] for sample in sample_indexes]
        predictions = self.model.predict(np.array(input_samples))

        print('')

        for (input, target, prediction) in zip(input_samples, target_samples, predictions):
            print('Input:      %s' % self.decypher(input))
            print('Target:     %s' % self.decypher(target))
            print('Prediction: %s\n' % self.decypher(prediction))

