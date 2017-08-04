import unittest
import pandas
import numpy as np

from model.topoml_util.Vectorizer import Vectorizer

TOPOLOGY_TRAINING_CSV = 'test_files/polygons.csv'
SOURCE_DATA = pandas.read_csv(TOPOLOGY_TRAINING_CSV)


class TestVectorizer(unittest.TestCase):
    def test_wkt_vectorize(self):
        brt_wkt = SOURCE_DATA['brt_wkt']
        osm_wkt = SOURCE_DATA['osm_wkt']
        vectorized = []
        for index in range(len(brt_wkt)):
            vectorized.append(Vectorizer.vectorize_wkt(brt_wkt[index], osm_wkt[index]))
        data_points = len(vectorized)
        (max_points, features) = max([array.shape for array in vectorized])
        np_array = np.zeros((data_points, max_points, features))
        for record_index, record in enumerate(vectorized):
            for point_index, point in enumerate(record):
                np_array[record_index][point_index] = point
        self.assertEqual(len(vectorized), 13)
        self.assertEqual(len(vectorized[0]), 44)
        self.assertEqual(len(vectorized[0][0]), 26)
