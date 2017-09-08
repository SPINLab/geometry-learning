import unittest
import numpy as np
from topoml_util.np_gaussian_2d_loss import np_rank2_gaussian_bivariate_loss


class TestNumpy2DGaussianLoss(unittest.TestCase):
    def test_2d_loss(self):
        vec_in = np.array([[1, 1, 0, 0, 0]])
        vec_out = vec_in
        loss1 = np_rank2_gaussian_bivariate_loss(vec_in, vec_out)
        vec_out = np.array([[1, 1, 5, 5, 0]])
        loss2 = np_rank2_gaussian_bivariate_loss(vec_in, vec_out)
        self.assertLess(loss1, loss2)

        vec_out = np.array([[1, 1, 5, -5, 0]])
        loss3 = np_rank2_gaussian_bivariate_loss(vec_in, vec_out)
        self.assertLess(loss1, loss3)
        self.assertEqual(loss2, loss3)
