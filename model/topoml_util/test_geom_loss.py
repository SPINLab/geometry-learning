import unittest

import numpy
import tensorflow as tf
from keras import backend as K
from .geom_loss import geom_loss

sess = tf.InteractiveSession()


class TestGeomLossFunction(unittest.TestCase):
    def test_zero_loss(self):
        tensor1 = numpy.array([[[1, 2, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        tensor2 = numpy.array([[[1, 2, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        K.print_tensor(tensor1)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 0.)

    def test_too_few_loss(self):
        tensor1 = numpy.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        tensor2 = numpy.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]], dtype=float)
        K.print_tensor(tensor1)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 0.)

    def test_geom_type_loss(self):
        tensor1 = numpy.array([[[1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = numpy.array([[[1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        K.print_tensor(tensor1)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 2)

    def test_geom_type_and_render_op_loss(self):
        tensor1 = numpy.array([[[1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]], dtype=float)
        tensor2 = numpy.array([[[1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]]], dtype=float)
        K.print_tensor(tensor1)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 4)

    def test_geom_one_coordinate_loss(self):
        tensor1 = numpy.array([[[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = numpy.array([[[1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        K.print_tensor(tensor1)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 1.)

    def test_geom_double_coordinate_loss(self):
        tensor1 = numpy.array([[[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        tensor2 = numpy.array([[[5, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=float)
        K.print_tensor(tensor1)
        loss = geom_loss(tensor1, tensor2).eval()
        self.assertEqual(loss, 2.)
