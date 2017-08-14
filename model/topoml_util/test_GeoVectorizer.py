import unittest
import pandas

from model.topoml_util.GeoVectorizer import GeoVectorizer, GEO_VECTOR_LEN, RENDER_INDEX, FULL_STOP_INDEX

TOPOLOGY_CSV = 'test_files/polygon_multipolygon.csv'
SOURCE_DATA = pandas.read_csv(TOPOLOGY_CSV)
brt_wkt = SOURCE_DATA['brt_wkt']
osm_wkt = SOURCE_DATA['osm_wkt']
target_wkt = SOURCE_DATA['intersection_wkt']


class TestVectorizer(unittest.TestCase):
    def test_max_points(self):
        max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
        self.assertEqual(max_points, 47)

    def test_wkt_vectorize_one_wkt(self):
        max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
        vectorized = []
        target_set = SOURCE_DATA['intersection_wkt']
        for index in range(len(target_set)):
            vectorized.append(GeoVectorizer.vectorize_wkt(target_set[index], max_points))
        self.assertEqual(len(target_set), len(brt_wkt))

    def test_wkt_vectorize_two_wkt(self):
        vectorized = []
        max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
        for index in range(len(brt_wkt)):
            vectorized.append(GeoVectorizer.vectorize_two_wkts(brt_wkt[index], osm_wkt[index], max_points))

        num_records = len(vectorized)
        num_points = len(vectorized[0])
        num_features = len(vectorized[0][0])

        self.assertEqual(num_records, 13)
        self.assertEqual(num_points, 47)
        self.assertEqual(num_features, GEO_VECTOR_LEN)

        for record in vectorized:
            points = [point for point in record if point[0] > 0]
            # Every first point should have a "render" code
            self.assertEqual(points[0][RENDER_INDEX], 1)
            # Every last point should have a "full stop" code
            self.assertEqual(points[-1][FULL_STOP_INDEX], 1)

    def test_decypher(self):
        self.maxDiff = None
        max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
        target_vector = GeoVectorizer.vectorize_wkt(target_wkt[0], max_points)
        decyphered = GeoVectorizer.decypher(target_vector)
        self.assertEqual(decyphered, target_wkt[0])
