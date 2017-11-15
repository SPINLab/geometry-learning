import unittest

from rasterio import features
from shapely import wkt


class TestRasterize(unittest.TestCase):
    def test_first(self):
        size = 20
        first = "POLYGON(({0} {0}, {0} -{0}, -{0} -{0}, -{0} {0}, {0} {0}))".format(size)
        geo_interfaces = [wkt.loads(first).__geo_interface__]
        raster = features.rasterize(geo_interfaces, out_shape=[255, 255])
        self.assertEqual(raster[100, 100], 1)
