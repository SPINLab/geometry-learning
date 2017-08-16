import numpy
from keras import backend as K
from keras.backend import floatx
from keras.losses import mse, categorical_crossentropy
from numpy.random import multivariate_normal

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
    num_samples = 100
    gaussian_errs = []
    # loss fn based on eq #26 of http://arxiv.org/abs/1308.0850.
    gaussian_loss = tf_2d_normal(y_true[:, :, 0],
                                 y_true[:, :, 1],
                                 y_pred[:, :, 0],
                                 y_pred[:, :, 1],
                                 y_pred[:, :, 2],
                                 y_pred[:, :, 3],
                                 y_pred[:, :, 4])
    geom_type_error = categorical_crossentropy(K.softmax(y_true[:, :, GEOM_TYPE_INDEX:RENDER_INDEX]),
                                               K.softmax(y_pred[:, :, GEOM_TYPE_INDEX:RENDER_INDEX]))
    render_error = categorical_crossentropy(K.softmax(y_true[:, :, RENDER_INDEX:]),
                                            K.softmax(y_pred[:, :, RENDER_INDEX:]))
    return gaussian_loss + geom_type_error + render_error


def point_to_gaussian_sample(geom_point, num_samples):
    #[mean_x, mean_y, sigma_x, sigma_y, rho] = geom_point[:, :, 0:5]
    sampled = multivariate_normal(mean=geom_point[:, :, 0:2],
                                  cov=[geom_point[:, :, 2:4], geom_point[:, :, 4:5]],
                                  size=num_samples)
    return sampled


def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2
    # eq 25
    z = ((norm1 / s1) + ((norm2 / s2) ** 2) -
         2 * ((rho * (norm1 * norm2)) / s1s2) ** 2)
    neg_rho = 1 - rho ** 2
    result = K.exp((-z / 2 * neg_rho))
    denom = 2 * numpy.pi * (s1s2 * (neg_rho ** (1/2)))
    result = (result / denom)
    return result

