import unittest
from keras import backend as K
import tensorflow as tf
import numpy as np
from sketch_rnn_model import tf_2d_normal

PRECISION = 6
sess = tf.InteractiveSession()


class TestSketchRnnLoss(unittest.TestCase):
    def test_2d_gaussian_zeros(self):
        target = np.array([[[0, 0]]], dtype=float)
        prediction = np.array([[[0, 0, 0, 0, 0]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, 1.1048509233685306, places=PRECISION)

    def test_2d_gaussian_small_mu_diff(self):
        target = np.array([[[5, 52]]], dtype=float)
        prediction = np.array([[[5 + 1e-6, 52 + 1e-6, 0, 0, 0]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, 1.1048509233706119, places=PRECISION)

    def test_2d_gaussian_small_sigma_diff(self):
        tensor_train = np.array([[[5, 52]]], dtype=float)
        tensor_predict = np.array([[[5, 52, 1e-6, 1e-6, 0]]], dtype=float)
        loss = tf_2d_normal(tensor_train, tensor_predict).eval()
        self.assertAlmostEqual(loss, 1.1048523660629765, places=PRECISION)

    def test_2d_gaussian_mu_ones(self):
        target = np.array([[[1, 1]]], dtype=float)
        prediction = np.array([[[1, 1, 1, 1, 0]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, 2.3829037437816121, places=PRECISION)

    def test_2d_gaussian_mu_minus_ones(self):
        target = np.array([[[1, 1]]], dtype=float)
        prediction = np.array([[[1, 1, -1, -1, 0]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, 2.3829037437816121, places=PRECISION)

    def test_2d_gaussian_ones(self):
        target = np.array([[[1, 1]]], dtype=float)
        prediction = np.array([[[1, 1, 1, 1, 1]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, 1.9491232946784192, places=PRECISION)

    def test_2d_gaussian_rho_one(self):
        target = np.array([[[1, 2]]], dtype=float)
        prediction = np.array([[[1, 2, 0, 0, 1]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, 1.9491232946784192, places=PRECISION)

    def test_2d_gaussian_rho_minus_one(self):
        target = np.array([[[1, 2]]], dtype=float)
        prediction = np.array([[[1, 2, 0, 0, -1]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, 1.9491232946784192, places=PRECISION)

    def test_2d_gaussian_rho_two(self):
        target = np.array([[[1, 2]]], dtype=float)
        prediction = np.array([[[1, 2, 0, 0, 2]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, 1.9491232946784192, places=PRECISION)

    def test_2d_gaussian_rho_minus_two(self):
        target = np.array([[[1, 2]]], dtype=float)
        prediction = np.array([[[1, 2, 0, 0, -2]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, 1.9491232946784192, places=PRECISION)

    def test_2d_gaussian_big_diff(self):
        target = np.array([[[5, 52]]], dtype=float)
        prediction = np.array([[[1, 2, 3, 4, 5]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, 16.11809565095832, places=PRECISION)

    def test_2d_gaussian_really_big_diff(self):
        target = np.array([[[5, 52]]], dtype=float)
        prediction = np.array([[[0, 0, 3, 4, 5]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, 16.11809565095832, places=PRECISION)

    def test_2d_gaussian_max_neg_rho(self):
        min_rho = -19.06  # This is about the limit of rho before geom_gaussian_loss returns NaN
        target = np.array([[[5, 52]]], dtype=float)
        prediction = np.array([[[5, 52, -1, -1, min_rho]]], dtype=float)
        args = np.append(target, prediction)
        loss = -K.log(tf_2d_normal(*args) + K.epsilon()).eval()
        self.assertAlmostEqual(loss, -18.505382378927028, places=PRECISION)