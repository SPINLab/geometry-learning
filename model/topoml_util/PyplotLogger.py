import pprint
from shapely.geometry import Point
from topoml_util.wkt2pyplot import wkt2pyplot
from keras.callbacks import Callback
import random
from datetime import datetime
import numpy as np

from topoml_util.GeoVectorizer import GeoVectorizer

pp = pprint.PrettyPrinter()


class DecypherAll(Callback):
    def __init__(self, gmm_size, sample_size=3, stdout=False):
        """
        Class constructor that instantiates with a few vital settings in order to decypher the output
        :param gmm_size: size as an integer of the gaussian mixture model
        :param sample_size: size as an integer of the number of samples to log
        :param stdout: boolean whether or not to log to stdout. Mixture models can have a lot of output.
        """
        super().__init__()
        self.gmm_size = gmm_size
        self.sample_size = sample_size
        self.stdout = stdout

    def on_epoch_end(self, epoch, logs=None):
        """
        Epochal logging function that outputs to a pyplot saved to a timestamped .png file
        :param epoch: automatically instantiated by Keras
        :param logs: automatically instantiated by Keras
        """
        random.seed(datetime.now())
        validation_samples = random.sample(range(len(self.validation_data[0])), self.sample_size)
        input_samples = [self.validation_data[0][sample] for sample in validation_samples]
        target_samples = [self.validation_data[1][sample] for sample in validation_samples]
        predictions = self.model.predict(np.array(input_samples))

        print('\nPlotting output for %i inputs, targets and predictions...' % len(predictions))

        for (input, target, prediction) in zip(input_samples, target_samples, predictions):
            timestamp = str(datetime.now()).replace(':', '.')

            if self.stdout:
                print('Input:')
                pp.pprint(input)
                print('Target:')
                pp.pprint(target)
                print('Prediction:')
                pp.pprint(prediction)

            input_polys = GeoVectorizer.decypher(input)
            target_points = [Point(point).wkt for point in
                             GeoVectorizer(gmm_size=self.gmm_size).decypher_gmm_geom(target, 10)]
            prediction_points = [Point(point).wkt for point in
                                 GeoVectorizer(gmm_size=self.gmm_size).decypher_gmm_geom(prediction, 10)]

            plt, fig, ax = wkt2pyplot(input_polys.split('\n'), target_points, prediction_points)
            plt.savefig('test_files/plt_' + timestamp + '.png')
            plt.close('all')
