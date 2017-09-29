import numpy as np
from keras import backend as K
from keras.backend import epsilon
from keras.losses import mse, categorical_crossentropy

from .GeoVectorizer import GEOM_TYPE_INDEX, RENDER_INDEX


def r3_geom_gaussian_loss(y_true, y_pred):
    # loss fn based on eq #26 of http://arxiv.org/abs/1308.0850.
    gaussian_loss = r3_bivariate_gaussian_loss(y_true, y_pred)
    geom_type_error = categorical_crossentropy(K.softmax(y_true[:, :, GEOM_TYPE_INDEX:RENDER_INDEX]),
                                               K.softmax(y_pred[:, :, GEOM_TYPE_INDEX:RENDER_INDEX]))
    render_error = categorical_crossentropy(K.softmax(y_true[:, :, RENDER_INDEX:]),
                                            K.softmax(y_pred[:, :, RENDER_INDEX:]))
    return gaussian_loss + geom_type_error + render_error


# Adapted to Keras from https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/model.py#L268
# Adapted version of the probability density function of
# https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
# augmented to negative log likelihood loss configuration
def r3_bivariate_gaussian_loss(true, pred):
    """
    Rank 3 bivariate gaussian loss function
    Returns results of eq # 24 of http://arxiv.org/abs/1308.0850
    :param true: truth values with at least [mu1, mu2, sigma1, sigma2, rho]
    :param pred: values predicted from a model with the same shape requirements as truth values
    :return: the log of the summed max likelihood
    """
    pdf = r3_bivariate_gaussian(pred, true)
    return K.log(K.sum(-K.log(pdf + epsilon())))  # → -∞ if pdf → ∞


def r3_bivariate_gaussian(true, pred):
    x_coord = true[:, :, 0]
    y_coord = true[:, :, 1]
    mu_x = pred[:, :, 0]
    mu_y = pred[:, :, 1]
    # exponentiate the sigmas and also make correlative rho between -1 and 1.
    # eq. # 21 and 22 of http://arxiv.org/abs/1308.0850
    # analogous to https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/model.py#L326
    sigma_x = K.exp(K.abs(pred[:, :, 2]))
    sigma_y = K.exp(K.abs(pred[:, :, 3]))
    rho = K.tanh(pred[:, :, 4]) * 0.1  # avoid drifting to -1 or 1 to prevent NaN
    norm1 = K.log(1 + K.abs(x_coord - mu_x))
    norm2 = K.log(1 + K.abs(y_coord - mu_y))
    variance_x = K.softplus(K.square(sigma_x))
    variance_y = K.softplus(K.square(sigma_y))
    s1s2 = K.softplus(sigma_x * sigma_y)  # very large if sigma_x and/or sigma_y are very large
    # eq 25 of http://arxiv.org/abs/1308.0850
    z = ((K.square(norm1) / variance_x) +
         (K.square(norm2) / variance_y) -
         (2 * rho * norm1 * norm2 / s1s2))  # z → -∞ if rho * norm1 * norm2 → ∞ and/or s1s2 → 0
    neg_rho = 1 - K.square(rho)  # → 0 if rho → {1, -1}
    numerator = K.exp(-z / (2 * neg_rho))  # → ∞ if z → -∞ and/or neg_rho → 0
    denominator = (2 * np.pi * s1s2 * K.sqrt(neg_rho)) + epsilon()  # → 0 if s1s2 → 0 and/or neg_rho → 0
    pdf = numerator / denominator  # → ∞ if denominator → 0 and/or if numerator → ∞
    return pdf


def r2_univariate_gaussian(true, pred):
    x = true[:, 0:1]
    mu = pred[:, 0:1]
    sigma = pred[:, 1:2]

    norm = K.log(1 + K.abs(x - mu))  # needs log of norm to counter large mu diffs
    variance = K.softplus(K.square(sigma))  # Softplus: prevent NaN on 0 sigma and converge to 0
    z = K.exp(-K.square(K.abs(norm)) / (2 * variance) + epsilon())  # z -> 0 if sigma

    # pdf -> 0 if sigma is very large or z -> 0; NaN if variance -> 0
    pdf = z / K.sqrt((2 * np.pi * variance) + epsilon())
    return pdf


def r3_univariate_gaussian(true, pred):
    x = true[:, :, 0:1]
    mu = pred[:, :, 0:1]
    sigma = pred[:, :, 1:2]

    norm = K.log(1 + K.abs(x - mu))  # needs log of norm to counter large mu diffs
    variance = K.softplus(K.square(sigma))
    z = K.exp(-K.square(K.abs(norm)) / (2 * variance) + epsilon())  # z -> 0 if sigma
    # pdf -> 0 if sigma is very large or z -> 0; NaN if variance -> 0
    pdf = z / K.sqrt((2 * np.pi * variance) + epsilon())
    return pdf


# Adapted to Keras from https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/model.py#L268
# Adapted version of the probability density function of
# https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
def r4_bivariate_gaussian(true, pred):
    """
    Rank 4 bivariate gaussian function
    Returns results of eq # 24 of http://arxiv.org/abs/1308.0850
    :param true: truth values with at least [mu1, mu2, sigma1, sigma2, rho]
    :param pred: values predicted from a model with the same shape requirements as truth values
    :return: the rank 4 probability density function
    """
    x_coord = true[:, :, :, 0]
    y_coord = true[:, :, :, 1]
    mu_x = pred[:, :, :, 0]
    mu_y = pred[:, :, :, 1]
    # exponentiate the sigmas and also make correlative rho between -1 and 1.
    # eq. # 21 and 22 of http://arxiv.org/abs/1308.0850
    # analogous to https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/model.py#L326
    sigma_x = K.exp(K.abs(pred[:, :, :, 2]))
    sigma_y = K.exp(K.abs(pred[:, :, :, 3]))
    rho = K.tanh(pred[:, :, :, 4]) * 0.1  # avoid drifting to -1 or 1 to prevent NaN
    norm1 = K.log(1 + K.abs(x_coord - mu_x))
    norm2 = K.log(1 + K.abs(y_coord - mu_y))
    variance_x = K.softplus(K.square(sigma_x))
    variance_y = K.softplus(K.square(sigma_y))
    s1s2 = K.softplus(sigma_x * sigma_y)  # very large if sigma_x and/or sigma_y are very large
    # eq 25 of http://arxiv.org/abs/1308.0850
    z = ((K.square(norm1) / variance_x) +
         (K.square(norm2) / variance_y) -
         (2 * rho * norm1 * norm2 / s1s2))  # z → -∞ if rho * norm1 * norm2 → ∞ and/or s1s2 → 0
    neg_rho = 1 - K.square(rho)  # → 0 if rho → {1, -1}
    numerator = K.exp(-z / (2 * neg_rho))  # → ∞ if z → -∞ and/or neg_rho → 0
    denominator = (2 * np.pi * s1s2 * K.sqrt(neg_rho)) + epsilon()  # → 0 if s1s2 → 0 and/or neg_rho → 0
    pdf = numerator / denominator  # → ∞ if denominator → 0 and/or if numerator → ∞
    return pdf


def r3_univariate_gaussian_loss(true, pred):
    pdf = r3_univariate_gaussian(true, pred)  # pdf -> 0 if sigma is very large or z -> 0
    return -K.log(pdf + epsilon())  # inf if pdf -> 0


def r2_univariate_gaussian_loss(true, pred):
    pdf = r2_univariate_gaussian(true, pred)
    return -K.log(pdf + epsilon())  # inf if pdf -> 0
