from keras.callbacks import Callback
import random
from datetime import datetime
import numpy as np


class CustomCallback(Callback):
    def __init__(self, decypher):
        super().__init__()
        self.decypher = decypher

    def on_epoch_end(self, epoch, logs=None):
        random.seed(datetime.now())
        validation_sample = random.sample(list(self.validation_data[0]), 5)
        predictions = self.model.predict(np.array(validation_sample))
        print('\nSome predictions on randomly sampled validation data:\n')
        for index in range(len(predictions)):
            print('Target:     %s' % self.decypher(validation_sample[index]))
            print('Prediction: %s\n' % self.decypher(predictions[index]))
