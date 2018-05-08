import unittest
import numpy as np

from topoml_util.GeoVectorizer import FULL_STOP_INDEX
from topoml_util import geom_scaler as gs

dummy_geom = np.zeros((1, 1, 13))

square = np.array([[
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
]])

normalized_square = np.array([[
    [-1., -1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [ 1., -1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [ 1.,  1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [-1.,  1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [-1., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
]])


class TestGeomScaler(unittest.TestCase):
    def test_localized_mean(self):
        means = gs.localized_mean(square)
        for mean in means[0, 0]:
            self.assertEqual(mean, 0.5)

    def test_scaling(self):
        scale = gs.scale(square)
        self.assertEqual(scale, 0.5)

    def test_transform(self):
        means = gs.localized_mean(square)
        scale = gs.scale(square)
        n_square = gs.transform(square, scale=scale)
        self.assertTrue((n_square == normalized_square).all())

    def test_dummy_geom(self):
        dummy_means = gs.localized_mean(dummy_geom)
        dummy_localized = gs.localized_normal(dummy_geom, dummy_means)
        self.assertEqual(dummy_localized[0, 0, FULL_STOP_INDEX], 1)

