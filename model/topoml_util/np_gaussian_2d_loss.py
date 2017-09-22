import numpy as np

epsilon = 1e-8


def softplus(x):
    return np.logaddexp(1.0, x)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Adapted version of the probability density function of
# https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
# augmented to negative log likelihood loss configuration
def np_r2_bivariate_gaussian_loss(true, pred):
    """Returns results of eq # 24 of http://arxiv.org/abs/1308.0850"""
    x_coord = true[:, 0]
    y_coord = true[:, 1]
    mu_x = pred[:, 0]
    mu_y = pred[:, 1]

    # exponentiate the sigmas and also make correlative rho between -1 and 1.
    # eq. # 21 and 22 of http://arxiv.org/abs/1308.0850
    # analogous to https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/model.py#L326
    sigma_x = np.exp(np.abs(pred[:, 2]))
    sigma_y = np.exp(np.abs(pred[:, 3]))
    rho = np.tanh(pred[:, 4])  # hardcode to avoid drifting to -1 or 1

    norm1 = np.log(1 + np.abs(x_coord - mu_x))
    norm2 = np.log(1 + np.abs(y_coord - mu_y))

    variance_x = softplus(np.square(sigma_x))
    variance_y = softplus(np.square(sigma_y))
    s1s2 = softplus(sigma_x * sigma_y)  # very large if sigma_x and/or sigma_y are very large

    # eq 25 of http://arxiv.org/abs/1308.0850
    z = ((np.square(norm1) / variance_x) +
         (np.square(norm2) / variance_y) -
         (2 * rho * norm1 * norm2 / s1s2))  # z → -∞ if rho * norm1 * norm2 → ∞ and/or s1s2 → 0
    neg_rho = 1 - np.square(rho)  # → 0 if rho → {1, -1}
    numerator = np.exp(-z / (2 * neg_rho))  # → ∞ if z → -∞ and/or neg_rho → 0
    denominator = (2 * np.pi * s1s2 * np.sqrt(neg_rho)) + epsilon  # → 0 if s1s2 → 0 and/or neg_rho → 0
    pdf = numerator / denominator  # → ∞ if denominator → 0 and/or if numerator → ∞
    return -np.log(pdf + epsilon)  # → -∞ if pdf → ∞


# Adapted version of the probability density function of
# https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
# augmented to negative log likelihood loss configuration
def np_r4_bivariate_gaussian_loss(true, pred):
    pdf = np_r4_bivariate_gaussian(true, pred)
    return -np.log(pdf + epsilon)  # → -∞ if pdf → ∞


def np_r4_bivariate_gaussian(true, pred):
    """Returns results of eq # 24 of http://arxiv.org/abs/1308.0850"""
    x_coord = true[:, :, :, 0]
    y_coord = true[:, :, :, 1]
    mu_x = pred[:, :, :, 0]
    mu_y = pred[:, :, :, 1]
    # exponentiate the sigmas and also make correlative rho between -1 and 1.
    # eq. # 21 and 22 of http://arxiv.org/abs/1308.0850
    # analogous to https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/model.py#L326
    sigma_x = np.exp(np.abs(pred[:, :, :, 2]))
    sigma_y = np.exp(np.abs(pred[:, :, :, 3]))
    rho = np.tanh(pred[:, :, :, 4]) * 0.1  # hardcode to avoid drifting to -1 or 1

    norm1 = np.log(1 + np.abs(x_coord - mu_x))
    norm2 = np.log(1 + np.abs(y_coord - mu_y))

    variance_x = softplus(np.square(sigma_x))
    variance_y = softplus(np.square(sigma_y))
    s1s2 = softplus(sigma_x * sigma_y)  # very large if sigma_x and/or sigma_y are very large
    # eq 25 of http://arxiv.org/abs/1308.0850
    z = ((np.square(norm1) / variance_x) +
         (np.square(norm2) / variance_y) -
         (2 * rho * norm1 * norm2 / s1s2))  # z → -∞ if rho * norm1 * norm2 → ∞ and/or s1s2 → 0
    neg_rho = 1 - np.square(rho)  # → 0 if rho → {1, -1}
    numerator = np.exp(-z / (2 * neg_rho))  # → ∞ if z → -∞ and/or neg_rho → 0
    denominator = (2 * np.pi * s1s2 * np.sqrt(neg_rho)) + epsilon  # → 0 if s1s2 → 0 and/or neg_rho → 0
    pdf = numerator / denominator  # → ∞ if denominator → 0 and/or if numerator → ∞
    return pdf
