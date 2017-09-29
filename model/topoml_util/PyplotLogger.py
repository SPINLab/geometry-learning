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
    def __init__(self, gmm_size, sample_size=3):
        super().__init__()
        self.gmm_size = gmm_size
        self.sample_size = sample_size

    def on_epoch_end(self, epoch, logs=None):
        random.seed(datetime.now())
        validation_samples = random.sample(range(len(self.validation_data[0])), self.sample_size)
        input_samples = [self.validation_data[0][sample] for sample in validation_samples]
        target_samples = [self.validation_data[1][sample] for sample in validation_samples]
        predictions = self.model.predict(np.array(input_samples))

        print('\nPlotting output for %i inputs, targets and predictions...' % len(predictions))

        for (input, target, prediction) in zip(input_samples, target_samples, predictions):
            timestamp = str(datetime.now()).replace(':', '.')

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

            plt = wkt2pyplot(input_polys.split('\n'), target_points, prediction_points)
            plt.savefig('test_files/plt_' + timestamp + '.png')
            plt.close('all')
