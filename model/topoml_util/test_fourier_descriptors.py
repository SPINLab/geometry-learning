import unittest
import numpy as np
from pyefd import elliptic_fourier_descriptors


class TestFourierDescriptors(unittest.TestCase):
    def test_same_descriptors(self):
        square1 = [[0, 0], [1, 0], [1, 0.5], [1, 1], [0, 1], [0, 0]]
        square2 = [[0, 0], [0.5, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        descriptors1 = elliptic_fourier_descriptors(square1)
        descriptors2 = elliptic_fourier_descriptors(square2)
        np.testing.assert_array_almost_equal(descriptors1, descriptors2)

    def test_different_descriptors(self):
        square1 = [[0, 0], [1, 0], [1, 0.5], [1, 1], [0, 1], [0, 0]]
        square2 = [[0, 0], [0.5, 0], [1, 0], [200, 300], [0, 1], [0, 0]]
        descriptors1 = elliptic_fourier_descriptors(square1)
        descriptors2 = elliptic_fourier_descriptors(square2)
        try:
            np.testing.assert_array_almost_equal(descriptors1, descriptors2)
        except Exception as e:
            self.assertEqual('Arrays are not almost equal to 6 decimals', e.args[0][1:42])
