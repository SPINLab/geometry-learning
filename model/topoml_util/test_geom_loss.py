import unittest
import array
import numpy as np
import tensorflow as tf
from keras import backend as K
from .geom_loss import geom_loss, geom_gaussian_loss, normal_2d_loss

sess = tf.InteractiveSession()


class TestGeomLossFunction(unittest.TestCase):
    def test_zero_loss(self):
        tensor1 = np.array([[[1, 2, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        tensor2 = np.array([[[1, 2, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 0.)

    def test_too_few_loss(self):
        tensor1 = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        tensor2 = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 0.)

    def test_geom_type_loss(self):
        tensor1 = np.array([[[1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        K.print_tensor(tensor1)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 2)

    def test_geom_type_and_render_op_loss(self):
        tensor1 = np.array([[[1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]]], dtype=float)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 4)

    def test_geom_one_coordinate_loss(self):
        tensor1 = np.array([[[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 1.)

    def test_geom_double_coordinate_loss(self):
        tensor1 = np.array([[[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[5, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 2.)


class TestGeomGaussianLoss(unittest.TestCase):
    def test_2d_guassian_relative_loss(self):
        tensor1 = np.array([[[1, 2]]], dtype=float)
        tensor2 = np.array([[[1, 2, 0, 0, 0]]], dtype=float)
        tensor3 = np.array([[[1, 2, 1, 1, 1]]], dtype=float)

        args1 = np.append(tensor1[:, :, 0:2], tensor2[:, :, 0:5])
        args2 = np.append(tensor1[:, :, 0:2], tensor3[:, :, 0:5])
        loss1 = normal_2d_loss(*args1).eval()
        loss2 = normal_2d_loss(*args2).eval()
        self.assertLess(loss1, loss2)

    def test_2d_gaussian_zero_diff(self):
        tensor1 = np.array([[[1, 2]]], dtype=float)
        tensor2 = np.array([[[1, 2, 1, 1, 1]]], dtype=float)
        args = np.append(tensor1[:, :, 0:2], tensor2[:, :, 0:5])

        normal = normal_2d_loss(*args).eval()
        self.assertAlmostEqual(normal, 3.40409623)

    def test_2d_gaussian_ones(self):
        tensor1 = np.array([[[1, 1]]], dtype=float)
        tensor2 = np.array([[[1, 1, 1, 1, 1]]], dtype=float)
        args = np.append(tensor1, tensor2)

        normal = normal_2d_loss(*args).eval()
        self.assertAlmostEqual(normal, 3.40409623)

    def test_2d_gaussian_zeros(self):
        tensor1 = np.array([[[0, 0]]], dtype=float)
        tensor2 = np.array([[[0, 0, 0, 0, 0]]], dtype=float)
        args = np.append(tensor1, tensor2)

        normal = normal_2d_loss(*args).eval()
        self.assertAlmostEqual(normal, 1.83787706)

    def test_2d_gaussian_small_diff(self):
        tensor1 = np.array([[[5, 52]]], dtype=float)
        tensor2 = np.array([[[5, 52, 1e-6, 1e-6, 1e-6]]], dtype=float)
        args = np.append(tensor1, tensor2)

        normal = normal_2d_loss(*args).eval()
        self.assertAlmostEqual(normal, 1.83787906)

    def test_2d_gaussian_big_diff(self):
        tensor1 = np.array([[[5, 52]]], dtype=float)
        tensor2 = np.array([[[1, 2, 3, 4, 5]]], dtype=float)
        args = np.append(tensor1, tensor2)

        normal = normal_2d_loss(*args).eval()
        self.assertAlmostEqual(normal, 4.53094815)

    def test_zero_loss(self):
        tensor1 = np.array([[[5, 52, 1, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        tensor2 = np.array([[[5, 52, 1, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 0.)

    def test_too_few_loss(self):
        tensor1 = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        tensor2 = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 0.)

    def test_geom_type_loss(self):
        tensor1 = np.array([[[1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 2, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 2)

    def test_geom_type_and_render_op_loss(self):
        tensor1 = np.array([[[1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 2, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 4)

    def test_geom_one_coordinate_loss(self):
        tensor1 = np.array([[[1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 1.)

    def test_geom_big_coordinate_loss(self):
        tensor1 = np.array([[[1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[5, 52, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertAlmostEqual(loss[0, 0], 2.17805637)

    def test_geom_big_inverted_coordinate_loss(self):
        tensor1 = np.array([[[5, 52, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertAlmostEqual(loss[0, 0], 3.19861807)

    def test_geom_sigma_rho_loss(self):
        tensor1 = np.array([[[5, 52, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = np.array([[[5, 52, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        loss = geom_gaussian_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 2.)
