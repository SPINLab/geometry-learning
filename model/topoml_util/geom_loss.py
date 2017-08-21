import numpy as np
from keras import backend as K
from keras.backend import tf, epsilon
from keras.losses import mse, categorical_crossentropy, kullback_leibler_divergence

from .GeoVectorizer import GEOM_TYPE_INDEX, RENDER_INDEX


def geom_loss(y_true, y_pred):
    print(y_true.shape)
    coordinate_error = mse(y_true[:, :, 0:2], y_pred[:, :, 0:2])
    geom_type_error = categorical_crossentropy(K.softmax(y_true[:, :, GEOM_TYPE_INDEX:RENDER_INDEX]),
                                               K.softmax(y_pred[:, :, GEOM_TYPE_INDEX:RENDER_INDEX]))
    render_error = categorical_crossentropy(K.softmax(y_true[:, :, RENDER_INDEX:]),
                                            K.softmax(y_pred[:, :, RENDER_INDEX:]))
    return coordinate_error + geom_type_error + render_error


def geom_gaussian_loss(y_true, y_pred):
    # loss fn based on eq #26 of http://arxiv.org/abs/1308.0850.
    gaussian_loss = gaussian_2d_loss(y_true, y_pred)
    geom_type_error = categorical_crossentropy(K.softmax(y_true[:, :, GEOM_TYPE_INDEX:RENDER_INDEX]),
                                               K.softmax(y_pred[:, :, GEOM_TYPE_INDEX:RENDER_INDEX]))
    render_error = categorical_crossentropy(K.softmax(y_true[:, :, RENDER_INDEX:]),
                                            K.softmax(y_pred[:, :, RENDER_INDEX:]))
    return gaussian_loss + geom_type_error + render_error


# Adapted to Keras From https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/model.py#L268
# Just a version of the probability density function of
# https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
def gaussian_2d_loss(true, pred):
    """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850"""
    x_coord = true[:, :, 0]
    y_coord = true[:, :, 1]
    mu_x = pred[:, :, 0]
    mu_y = pred[:, :, 1]
    sigma_x = pred[:, :, 2]
    sigma_y = pred[:, :, 3]
    rho = pred[:, :, 4]

    norm1 = K.abs(x_coord - mu_x)
    norm2 = K.abs(y_coord - mu_y)

    # exponentiate the sigmas and also make correlative rho between -1 and 1.
    # eq. # 21 and 22 of http://arxiv.org/abs/1308.0850
    # analogous to https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/model.py#L326
    sigma_x = K.exp(K.abs(sigma_x))
    sigma_y = K.exp(K.abs(sigma_y))
    rho = K.tanh(rho)
    s1s2 = sigma_x * sigma_y  # very large if sigma_x and/or sigma_y are very large

    # eq 25 of http://arxiv.org/abs/1308.0850
    z = ((K.square(tf.div(norm1, sigma_x))) +
         (K.square(tf.div(norm2, sigma_y))) -
         (2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)))
    neg_rho = 1 - (K.square(rho))  # never very large
    result = K.exp(tf.div(-z, (2 * neg_rho)))
    denom = 2 * np.pi * (s1s2 * (K.sqrt(neg_rho)))  # very small if s1s2 and/or neg_rho are very large
    result = tf.div(result, denom)  # very large if denom is very small
    return -K.log(result + epsilon())  # negative if result is very large


def gaussian_1d_loss(target, prediction):
    x = target[:, :, 0:1]
    mu = prediction[:, :, 0:1]
    sigma = K.exp(K.abs(prediction[:, :, 1:2]))
    z = K.exp(-((K.abs(x - mu) / 2 * sigma) ** 2))
    pdf = z / K.sqrt(2 * np.pi * sigma ** 2)
    return -K.log(pdf)
