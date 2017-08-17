import numpy as np
from keras import backend as K
from keras.backend import floatx, tf, epsilon
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
    # for each geom_point, sample a set of 100 gaussian samples
    # loss fn based on eq #26 of http://arxiv.org/abs/1308.0850.
    gaussian_loss = normal_2d_loss(y_true[:, :, 0],
                                   y_true[:, :, 1],
                                   y_pred[:, :, 0],
                                   y_pred[:, :, 1],
                                   y_pred[:, :, 2],
                                   y_pred[:, :, 3],
                                   y_pred[:, :, 4])

    gaussian_loss = K.sigmoid(gaussian_loss)
    # kl_loss = kullback_leibler_divergence(y_true[:, :, 0:5], y_pred[:, :, 0:5])
    geom_type_error = categorical_crossentropy(K.softmax(y_true[:, :, GEOM_TYPE_INDEX:RENDER_INDEX]),
                                               K.softmax(y_pred[:, :, GEOM_TYPE_INDEX:RENDER_INDEX]))
    render_error = categorical_crossentropy(K.softmax(y_true[:, :, RENDER_INDEX:]),
                                            K.softmax(y_pred[:, :, RENDER_INDEX:]))
    return gaussian_loss + geom_type_error + render_error


# Adapted to Keras From https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/model.py#L268
# Just a version of the probability density function of
# https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
def normal_2d_loss(x1, x2, mu1, mu2, s1, s2, rho):
    """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850"""
    norm1 = x1 - mu1
    norm2 = x2 - mu2

    # exponentiate the sigmas and also make correlative rho between -1 and 1.
    # eq. # 21 and 22 of http://arxiv.org/abs/1308.0850
    # analogous to https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/model.py#L326
    s1 = K.exp(s1)
    s2 = K.exp(s2)
    rho = K.tanh(rho)

    s1s2 = (s1 * s2)

    # eq 25 of http://arxiv.org/abs/1308.0850
    z = ((norm1 / (s1 ** 2)) + (norm2 / (s2 ** 2)) -
         2 * ((rho * (norm1 * norm2)) / s1s2))
    neg_rho = 1 - (rho ** 2)
    result = K.exp(-z / 2 * neg_rho)
    denom = 2 * np.pi * (s1s2 * (neg_rho ** 0.5))
    result = (result / denom)
    return -K.log(K.sum(result))

