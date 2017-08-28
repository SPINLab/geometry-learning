# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sketch-RNN Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras import backend as K

import random

# internal imports

import numpy as np
import tensorflow as tf


# NB: the below are inner functions, not methods of Model
def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
    # exponentiate the sigmas and also make corr between -1 and 1.
    print_op = tf.Print()
    s1 = tf.exp(s1)
    s2 = tf.exp(s2)
    rho = tf.tanh(rho)

    norm1 = tf.subtract(x1, mu1)
    norm2 = tf.subtract(x2, mu2)
    s1s2 = tf.multiply(s1, s2)
    # eq 25
    z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
         2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
    neg_rho = 1 - tf.square(rho)
    result = tf.exp(tf.div(-z, 2 * neg_rho))
    denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
    result = tf.div(result, denom)
    return result


def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                 z_pen_logits, x1_data, x2_data, pen_data):
    """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
    # This represents the L_R only (i.e. does not include the KL loss term).

    result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                           z_corr)
    epsilon = 1e-6
    # result1 is the loss wrt pen offset (L_s in equation 9 of
    # https://arxiv.org/pdf/1704.03477.pdf)
    result1 = tf.multiply(result0, z_pi)
    result1 = tf.reduce_sum(result1, 1, keep_dims=True)
    result1 = -tf.log(result1 + epsilon)  # avoid log(0)

    fs = 1.0 - pen_data[:, 2]  # use training data for this
    fs = tf.reshape(fs, [-1, 1])
    # Zero out loss terms beyond N_s, the last actual stroke
    result1 = tf.multiply(result1, fs)

    # result2: loss wrt pen state, (L_p in equation 9)
    result2 = tf.nn.softmax_cross_entropy_with_logits(
        labels=pen_data, logits=z_pen_logits)
    result2 = tf.reshape(result2, [-1, 1])
    if not self.hps.is_training:  # eval mode, mask eos columns
        result2 = tf.multiply(result2, fs)

    result = result1 + result2
    return result


# below is where we need to do MDN (Mixture Density Network) splitting of
# distribution params
def get_mixture_coef(output):
    """Returns the tf slices containing mdn dist params."""
    # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
    z = output
    z_pen_logits = z[:, 0:3]  # pen states
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

    # process output z's into MDN paramters

    # softmax all the pi's and pen states:
    z_pi = tf.nn.softmax(z_pi)
    z_pen = tf.nn.softmax(z_pen_logits)

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = tf.exp(z_sigma1)
    z_sigma2 = tf.exp(z_sigma2)
    z_corr = tf.tanh(z_corr)

    r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
    return r
