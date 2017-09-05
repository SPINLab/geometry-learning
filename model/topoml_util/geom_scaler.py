import numpy as np
from .GeoVectorizer import FULL_STOP_INDEX


def localized_normal(vectors, scale=1e4):
    (data_points, max_points, GEO_VECTOR_LEN) = vectors.shape

    for data_point in vectors:
        full_stop_point_index = data_point[:, FULL_STOP_INDEX].tolist().index(1)
        # Take the mean of all non-null points for localized origin
        geom_mean = np.mean(data_point[0:full_stop_point_index + 1, 0:2], axis=0, keepdims=True)
        # Repeat base for each point
        geom_mean = np.repeat(geom_mean, full_stop_point_index + 1, axis=0)
        zeros = np.zeros((max_points - 1 - full_stop_point_index, 2))
        correction = np.append(geom_mean, zeros, axis=0)
        data_point[:, 0:2] = (data_point[:, 0:2] - correction) * scale

    return vectors
