import unittest
import numpy as np
from topoml_util.GeoVectorizer import FULL_STOP_INDEX

from topoml_util.geom_scaler import localized_normal, localized_mean

input_geom = np.array([[
  [6.7231583, 52.147836, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
  [6.72318, 52.1477406, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
  [6.7232779, 52.147749, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
  [6.7232562, 52.1478444, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.], 
  [6.7231583, 52.147836, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.], 
  [6.7231633, 52.1478309, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
  [6.7232509, 52.1478377, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
  [6.7232692, 52.1477484, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
  [6.7231819, 52.1477415, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
  [6.7231633, 52.1478309, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., ],
  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ]
]])

dummy_geom = np.array([[
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
]])

means = localized_mean(input_geom)
scaled = localized_normal(input_geom, means)


class TestGeomScaler(unittest.TestCase):
    def test_scaling(self):
        self.assertAlmostEqual(scaled[0, 0, 0], -0.4763)
        self.assertAlmostEqual(scaled[0, 0, 1], 0.3646)
        self.assertAlmostEqual(scaled[0, 9, 0], -0.4263)
        self.assertAlmostEqual(scaled[0, 9, 1], 0.3136)

    def test_localized_mean(self):
        self.assertAlmostEqual(means[0, 0, 0], 6.7232059)
        self.assertAlmostEqual(means[0, 0, 1], 52.1477995)
        corrected = input_geom
        corrected[:, 0:10, 0:2] -= means
        corrected[:, 0:10, 0:2] *= 1e4
        equal = np.equal(corrected, scaled)
        self.assertTrue(np.array_equal(corrected, scaled))

    def test_dummy_geom(self):
        dummy_means = localized_mean(dummy_geom)
        dummy_localized = localized_normal(dummy_geom, dummy_means)
        self.assertEqual(dummy_localized[0, 0, FULL_STOP_INDEX], 1)

