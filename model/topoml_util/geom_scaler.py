import numpy as np
from .GeoVectorizer import FULL_STOP_INDEX


def scale(vectors):
    means = localized_mean(vectors)
    min_maxs = []

    for index, data_point in enumerate(vectors):
        full_stop_point = data_point[:, FULL_STOP_INDEX].tolist()

        try:
            full_stop_point_index = full_stop_point.index(1)
            min_maxs.append([
                    np.min(data_point[..., :full_stop_point_index, :2] - means[index]),
                    np.max(data_point[..., :full_stop_point_index, :2] - means[index])
                    ])
        except Exception as e:  # if a dummy point is encountered
            min_maxs.append([0, 0])
            continue

    return np.std(min_maxs)


def transform(vectors, scale=None):
    localized = np.copy(vectors)
    means = localized_mean(vectors)

    for index, data_point in enumerate(localized):
        full_stop_point = data_point[:, FULL_STOP_INDEX].tolist()

        try:
            full_stop_point_index = full_stop_point.index(1)
        except Exception as e:  # if a dummy point is encountered
            continue

        data_point[..., :full_stop_point_index + 1, :2] -= means[index]
        data_point[..., :full_stop_point_index + 1, :2] /= scale

    return localized


def localized_mean(vectors):
    geom_means = []
    for data_point in vectors:
        full_stop_point = data_point[:, FULL_STOP_INDEX].tolist()

        try:
            full_stop_point_index = full_stop_point.index(1)
        except Exception as e:  # if a dummy point is encountered
            geom_means.append([[[0, 0]]])
            continue

        # Take the mean of all non-null points for localized origin
        geom_mean = np.mean(data_point[0:full_stop_point_index, 0:2], axis=0, keepdims=True)
        geom_means.append(geom_mean)

    return np.array(geom_means)


def localized_normal(vectors, means, scale=1e4):
    localized = np.copy(vectors)
    data_points, max_points, GEO_VECTOR_LEN = localized.shape

    for index, data_point in enumerate(localized):
        full_stop_point = data_point[:, FULL_STOP_INDEX].tolist()
        try:
            full_stop_point_index = full_stop_point.index(1)
        except Exception as e:  # 1 is not in list, it is an empty geometry
            data_point[0, FULL_STOP_INDEX] = 1
            continue

        # repeat for each non-null node in the geometry
        geom_mean = np.repeat(means[index], full_stop_point_index + 1, axis=0)
        zeros = np.zeros((max_points - 1 - full_stop_point_index, 2))
        correction = np.append(geom_mean, zeros, axis=0)
        data_point[:, 0:2] = (data_point[:, 0:2] - correction) * scale

    return localized
