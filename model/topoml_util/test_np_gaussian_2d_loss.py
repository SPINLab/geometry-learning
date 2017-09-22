import unittest
import numpy as np
from np_gaussian_2d_loss import np_r2_bivariate_gaussian_loss, np_r4_bivariate_gaussian, softmax, epsilon, \
    np_r4_bivariate_gaussian_loss


class TestNumpy2DGaussianLoss(unittest.TestCase):
    def test_r2_2d_loss(self):
        vec_in = np.array([[1, 1, 0, 0, 0]])
        vec_out = vec_in
        loss1 = np_r2_bivariate_gaussian_loss(vec_in, vec_out)
        vec_out = np.array([[1, 1, 5, 5, 0]])
        loss2 = np_r2_bivariate_gaussian_loss(vec_in, vec_out)
        self.assertLess(loss1, loss2)

        vec_out = np.array([[1, 1, 5, -5, 0]])
        loss3 = np_r2_bivariate_gaussian_loss(vec_in, vec_out)
        self.assertLess(loss1, loss3)
        self.assertEqual(loss2, loss3)

    def test_r4_bivariate_gaussian_loss(self):
        vec_in = np.array([[[[1, 1, 0, 0, 0, 0]]]])
        vec_out = vec_in
        loss1 = np_r4_bivariate_gaussian_loss(vec_in, vec_out)
        vec_out = np.array([[[[1, 1, 5, 5, 0, 0]]]])
        loss2 = np_r4_bivariate_gaussian_loss(vec_in, vec_out)
        self.assertLess(loss1, loss2)

        vec_out = np.array([[[[1, 1, 5, -5, 0, 0]]]])
        loss3 = np_r4_bivariate_gaussian_loss(vec_in, vec_out)
        self.assertLess(loss1, loss3)
        self.assertEqual(loss2, loss3)

    def test_r4_bivariate_gmm_zeros_loss(self):
        vec_in = np.array([[[[0, 0, 0, 0, 0, 0]]]])
        vec_in = np.repeat(vec_in, 6, axis=2)  # 6 gaussian mixture components
        pi_index = 5
        pi_weights = softmax(vec_in[:, :, :, pi_index])
        vec_out = vec_in
        loss1 = np_r4_bivariate_gaussian(vec_in, vec_out)
        loss1 = loss1 * pi_weights
        gmm_loss1 = np.sum(-np.log(loss1 + epsilon), keepdims=True)

        vec_out = np.array([[[[1, 1, 5, 5, 0, 0]]]])
        vec_out = np.repeat(vec_out, 6, axis=2)
        pi_weights = softmax(vec_out[:, :, :, pi_index])
        loss2 = np_r4_bivariate_gaussian(vec_in, vec_out) * pi_weights
        gmm_loss2 = np.sum(-np.log(loss2 + epsilon), keepdims=True)
        self.assertLess(gmm_loss1[0, 0, 0], gmm_loss2[0, 0, 0])
