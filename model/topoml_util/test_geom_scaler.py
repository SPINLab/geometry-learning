import unittest

import numpy as np

from topoml_util import geom_scaler as gs

# noinspection PyUnresolvedReferences
dummy_geom = np.zeros((1, 1, 5))

square = np.array([[
    [0., 0., 1., 0., 0.],
    [1., 0., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [0., 1., 1., 0., 0.],
    [0., 0., 0., 0., 1.],
]])

square_duplicate_nodes = np.array([[
    [0., 0., 1., 0., 0.],
    [1., 0., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [0., 1., 1., 0., 0.],
    [0., 0., 0., 0., 1.],
]])

rectangle = np.array([[
    [0., 0., 1., 0., 0.],
    [1., 0., 1., 0., 0.],
    [1., 2., 1., 0., 0.],
    [0., 2., 1., 0., 0.],
    [0., 0., 0., 0., 1.],
]])

normalized_square = np.array([[
    [-1., -1., 1., 0., 0.],
    [ 1., -1., 1., 0., 0.],
    [ 1.,  1., 1., 0., 0.],
    [-1.,  1., 1., 0., 0.],
    [-1., -1., 0., 0., 1.],
]])


class TestGeomScaler(unittest.TestCase):
    def test_localized_mean(self):
        means = gs.localized_mean(square)
        for mean in means[0]:
            self.assertTrue((mean == 0.5).all())

    def test_localized_mean_rectangle(self):
        means = gs.localized_mean(rectangle)
        self.assertEqual(means[0, 0, 0], 0.5)
        self.assertEqual(means[0, 0, 1], 1)

    def test_localized_mean_dup_nodes(self):
        means = gs.localized_mean(square_duplicate_nodes)
        self.assertTrue((means == 0.75).all())

    def test_scaling_square(self):
        scale = gs.scale(square)
        self.assertEqual(scale, 0.5)

    def test_scaling_square_dup_nodes(self):
        scale = gs.scale(square_duplicate_nodes)
        self.assertEqual(scale, 0.5)

    def test_transform(self):
        # scaled_square = square[0] * 2
        # scaled_square[4, 12] = 1.
        scale = gs.scale(square)
        n_square = gs.transform(square, scale=scale)
        self.assertTrue((n_square == normalized_square).all())
        coords = [geom[:, :2].flatten() for geom in n_square]
        coords = [item for sublist in coords for item in sublist]
        std = np.std(coords)
        self.assertAlmostEqual(std, 1., 1)

    def test_upsized_transform(self):
        square_0 = square[0] * 2
        square_0[:4, 2] = 1.
        square_0[4, 4] = 1.
        scale = gs.scale([square_0])
        n_square = gs.transform([square_0], scale=scale)
        self.assertTrue((n_square == normalized_square).all())
        coords = [geom[:, :2].flatten() for geom in n_square]
        coords = [item for sublist in coords for item in sublist]
        std = np.std(coords)
        self.assertAlmostEqual(std, 1., 1)
