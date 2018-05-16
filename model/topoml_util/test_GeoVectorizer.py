import unittest

import numpy as np
import pandas
from GeoVectorizer import GeoVectorizer, GEO_VECTOR_LEN
from shapely import wkt as wktreader

TOPOLOGY_CSV = 'test_files/polygon_multipolygon.csv'
SOURCE_DATA = pandas.read_csv(TOPOLOGY_CSV)
brt_wkt = SOURCE_DATA['brt_wkt']
osm_wkt = SOURCE_DATA['osm_wkt']
target_wkt = SOURCE_DATA['intersection_wkt']

input_geom = np.array([
    [0., 0., 1., 0., 0.],
    [0., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 0., 1., 0., 0.],
    [0., 0., 0., 1., 0.],
    [0., 0., 1., 0., 0.],
    [0., -1., 1., 0., 0.],
    [-1., -1., 1., 0., 0.],
    [-1., 0., 1., 0., 0.],
    [0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0.]
])

output_geom = np.array([
    [0.0, 0.00, 1., 0., 0.],
    [0.0, 0.25, 1., 0., 0.],
    [0.0, 0.50, 1., 0., 0.],
    [0.0, 0.75, 1., 0., 0.],
    [0.0, 1.00, 1., 0., 0.],
    [0.25, 1.0, 1., 0., 0.],
    [0.50, 1.0, 1., 0., 0.],
    [1.0, 1.00, 1., 0., 0.],
    [1.0, 0.50, 1., 0., 0.],
    [1.0, 0.00, 1., 0., 0.],
    [0.5, 0.00, 1., 0., 0.],
    [0.0, 0.00, 0., 1., 0.],
    [0.0, 0.00, 1., 0., 0.],
    [0.0, -0.5, 1., 0., 0.],
    [0.0, -1.0, 1., 0., 0.],
    [-0.5, -1., 1., 0., 0.],
    [-1., -1.0, 1., 0., 0.],
    [-1., -0.5, 1., 0., 0.],
    [-1., 0.00, 1., 0., 0.],
    [-0.5, 0.0, 1., 0., 0.],
    [0.00, 0.0, 0., 0., 1.],
    [0.00, 0.0, 0., 0., 0.]
])

non_empty_geom_collection = 'GEOMETRYCOLLECTION(LINESTRING(1 1, 3 5),POLYGON((-1 -1, -1 -5, -5 -5, -5 -1, -1 -1)))'


class TestVectorizer(unittest.TestCase):
    def test_max_points(self):
        max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
        self.assertEqual(max_points, 159)

    # def test_interpolate(self):
    #     interpolated = GeoVectorizer.interpolate(input_geom, len(input_geom) * 2)
    #     for index, _ in enumerate(interpolated):
    #         result = list(interpolated[index])
    #         expected = list(output_geom[index])
    #         self.assertListEqual(result, expected, msg='Lists differ at index %i' % index)

    def test_vectorize_one_wkt(self):
        max_points = 20
        input_set = SOURCE_DATA['intersection_wkt']
        vectorized = []
        for index in range(len(input_set)):
            vectorized.append(GeoVectorizer.vectorize_wkt(input_set[index], max_points, simplify=True))
        self.assertEqual(len(input_set), len(brt_wkt))
        self.assertEqual(vectorized[0].shape, (19, GEO_VECTOR_LEN))
        self.assertEqual(vectorized[1].shape, (1, GEO_VECTOR_LEN))

    def test_fixed_size(self):
        max_points = 20
        input_set = SOURCE_DATA['intersection_wkt']
        vectorized = [GeoVectorizer.vectorize_wkt(wkt, max_points, simplify=True, fixed_size=True) for wkt in input_set]
        self.assertEqual(np.array(vectorized).shape, (input_set.size, 20, GEO_VECTOR_LEN))

    def test_non_empty_geom_coll(self):
        with self.assertRaises(ValueError):
            GeoVectorizer.vectorize_wkt(non_empty_geom_collection, 100)

    def test_point(self):
        point_matrix = GeoVectorizer.vectorize_wkt('POINT(12 14)', 5)
        self.assertEqual(point_matrix.shape, (1, GEO_VECTOR_LEN))

    def test_unsupported_geom(self):
        # Since
        with self.assertRaises(Exception):
            GeoVectorizer.vectorize_wkt(
                'TEST_FOR_UNKNOWN_GEOM_TYPE ((10 10, 20 20, 10 40),(40 40, 30 30, 40 20, 30 10))', 16)

    def test_vectorize_big_multipolygon(self):
        with open('test_files/big_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            max_points = GeoVectorizer.max_points([wkt])
            vectorized = GeoVectorizer.vectorize_wkt(wkt, max_points)
            self.assertEqual((144, GEO_VECTOR_LEN), vectorized.shape)

    def test_simplify_multipolygon_gt_max_points(self):
        with open('test_files/multipart_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            max_points = 20
            vectorized = GeoVectorizer.vectorize_wkt(wkt, max_points, simplify=True)
            self.assertEqual((20, GEO_VECTOR_LEN), vectorized.shape)

    def test_multipolygon_exceed_max_points(self):
        with open('test_files/multipart_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            max_points = 20
            with self.assertRaises(Exception):
                GeoVectorizer.vectorize_wkt(wkt, max_points)

    def test_polygon_exceed_max_points(self):
        with open('test_files/multipart_multipolygon_wkt.txt', 'r') as file:
            wkt = file.read()
            shape = wktreader.loads(wkt)
            geom = shape.geoms[0]
            max_points = 20
            with self.assertRaises(Exception):
                GeoVectorizer.vectorize_wkt(geom.wkt, max_points)
