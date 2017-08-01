import unittest
from .Vectorize import Vectorize


class TestVectorize(unittest.TestCase):
    def test_vectorize_point(self):
        wkt = "POINT (0 1)"
        coords = Vectorize.vectorize(wkt)
        self.assertEqual(coords, [(0.0, 1.0)])

    def test_vectorize_polygon(self):
        wkt = "POLYGON((24 37,30 37,30 33,24 33,24 37))"
        geo_shape = Vectorize.vectorize(wkt)
        self.assertEqual(geo_shape, [(24.0, 37.0), (30.0, 37.0), (30.0, 33.0), (24.0, 33.0), (24.0, 37.0)])
