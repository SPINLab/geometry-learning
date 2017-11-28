from shapely.wkt import loads
from shapely import geometry
import numpy as np
import math

# TODO: refactor GEOMETRY_TYPES to use shapely.geometry.base.GEOMETRY_TYPE
GEOMETRY_TYPES = ["GeometryCollection", "Point", "LineString", "Polygon", "MultiPoint", "MultiLineString",
                  "MultiPolygon", "Geometry"]
X_INDEX = 0  # the X coordinate position
Y_INDEX = 1  # the Y coordinate position
GEOM_TYPE_INDEX = 5  # Start index of the geometry type
GEOM_TYPE_LEN = 8  # 8 positions in the one-hot encoding for the geometry type
RENDER_INDEX = GEOM_TYPE_INDEX + GEOM_TYPE_LEN  # Render index start
RENDER_LEN = 3  # Render one-hot vector length
ONE_HOT_LEN = GEOM_TYPE_LEN + RENDER_LEN  # Length of the one-hot encoded part
STOP_INDEX = RENDER_INDEX + 1  # Stop index for the first geometry. A second one follows
FULL_STOP_INDEX = STOP_INDEX + 1  # Full stop index. No more points to follow
GEO_VECTOR_LEN = FULL_STOP_INDEX + 1  # The length needed to describe the features of a geometry point

action_types = ["render", "stop", "full stop"]
wkt_start = {
    "GeometryCollection": " EMPTY",
    "Polygon": "((",
    "MultiPolygon": "(((",
    "Point": "(",
    "LineString": "(",
    "MultiPoint": "((",
    "MultiLineString": "((",
    "Geometry": "("
}
wkt_end = {
    "GeometryCollection": "",
    "Polygon": "))",
    "MultiPolygon": ")))",
    "Point": ")",
    "LineString": ")",
    "MultiPoint": "))",
    "MultiLineString": "))",
    "Geometry": ")"
}


class GeoVectorizer:
    def __init__(self, gmm_size=None):
        self.gmm_size = gmm_size

    @staticmethod
    def softplus(x):
        """Compute softplus values for each sets of values in x."""
        return np.logaddexp(1.0, x)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of values in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def max_points(*wkt_sets):
        """
        Determines the maximum summed size (length) of elements in an arbitrary length 1d array of well-known-text
        geometries
        :param wkt_sets: arbitrary length array of 1d arrays containing well-known-text geometry entries
        :return: scalar integer representing the longest set of points length
        """
        max_points = 0

        for wkts in zip(*wkt_sets):
            number_of_points = sum([GeoVectorizer.num_points_from_wkt(wkt) for wkt in wkts])
            if number_of_points > max_points:
                max_points = number_of_points

        return max_points

    @staticmethod
    def interpolate(input_vectors, max_points):
        if len(input_vectors) > max_points:
            raise ValueError('The length of the input vector is already larger than max points', max_points)

        interpolated = np.copy(input_vectors)  # just in case len(input_vectors) == max_points

        index = 0
        while max_points > len(interpolated) > index:
            if not interpolated[index, RENDER_INDEX]:  # i.e. no STOP or FULL stop
                index += 1
                continue
            halfway = np.mean(interpolated[index:index + 2, 0:2], axis=0, keepdims=True)
            halfway = np.append(halfway, interpolated[index, 2:])
            interpolated = np.insert(interpolated, index + 1, halfway, axis=0)
            index += 2

        if len(interpolated) < max_points:
            interpolated = GeoVectorizer.interpolate(interpolated, max_points)

        return interpolated

    @staticmethod
    def num_points_from_wkt(wkt):
        shape = loads(wkt)
        if shape.geom_type == 'Polygon':
            number_of_points = len(shape.exterior.coords)
        elif shape.geom_type == 'MultiPolygon':
            number_of_points = len(shape.geoms[0].exterior.coords)
        else:
            raise ValueError("Don't know how to get the number of points from geometry type %s" % (
                shape.geom_type))

        return number_of_points

    @staticmethod
    def vectorize_wkt(wkt, number_of_points, simplify=False):
        """
        Convert wkt geometry to numpy array of real values
        :param wkt: the geometry as wkt string
        :param number_of_points: the size of the output first dimension: the maximum number of points in the two
                                    combined wkt points
        :param simplify: optional, selecting reduction of points if wkt points exceeds max_points
        :return vectors: a 2d numpy array as vectorized representation of the input geometry
        """
        shape = loads(wkt)
        vector = np.zeros((number_of_points, GEO_VECTOR_LEN))

        if shape.geom_type == 'Polygon':
            if len(shape.exterior.coords) > number_of_points:
                if not simplify:
                    raise ValueError('The number of points in the geometry exceeds the max points in the output '
                                     'specification, but the reduce_points parameter was set to False. Please set the '
                                     'reduce_points parameter to True to reduce the number of points, or increase the '
                                     'max points.')

            vector = GeoVectorizer.vectorize_points(shape.exterior.coords, vector,
                                                    shape.geom_type, is_last=True, simplify=simplify)
        elif shape.geom_type == 'MultiPolygon':
            total_points = sum([len(geom.exterior.coords) for geom in shape.geoms])

            if total_points > number_of_points:
                if not simplify:
                    raise ValueError('The number of points in the geometry exceeds the max points in the output '
                                     'specification, but the reduce_points parameter was set to False. Please set the '
                                     'reduce_points parameter to True to reduce the number of points, or increase the '
                                     'max points.')

            log_tolerance = -10  # Log scale
            tolerance = math.pow(10, log_tolerance)

            while total_points > number_of_points:
                shape = shape.simplify(tolerance)
                if shape.geom_type.startswith('Multi'):  # The geometry type can change due to simplification
                    total_points = sum([len(geom.exterior.coords) for geom in shape.geoms])
                else:
                    total_points = len(shape.exterior.coords)
                log_tolerance += 0.1
                tolerance = math.pow(10, log_tolerance)

            if shape.geom_type.startswith('Multi'):  # The geometry type can change due to simplification
                vector = np.concatenate([GeoVectorizer.vectorize_points(
                    points=geom.exterior.coords,
                    vector=np.zeros((len(geom.exterior.coords), GEO_VECTOR_LEN)),
                    geom_type=geom.geom_type
                ) for geom in shape.geoms], axis=0)
            else:
                vector = GeoVectorizer.vectorize_points(
                    points=shape.exterior.coords,
                    vector=np.zeros((len(shape.exterior.coords), GEO_VECTOR_LEN)),
                    geom_type=shape.geom_type
                )

            vector[total_points - 1, STOP_INDEX] = 0
            vector = np.append(vector, np.zeros((number_of_points - total_points, GEO_VECTOR_LEN)), axis=0)
            vector[total_points - 1:, FULL_STOP_INDEX] = 1  # Manually set full stop bits

            return vector

        elif shape.geom_type == 'GeometryCollection':
            if len(shape.geoms) > 0:  # not GEOMETRYCOLLECTION EMPTY
                raise ValueError("Don't know how to process non-empty GeometryCollection type")

        elif shape.geom_type == 'Point':
            vector = GeoVectorizer.vectorize_points(shape.coords, vector,
                                                    shape.geom_type, is_last=True, simplify=simplify)
        else:
            raise ValueError("Don't know how to get the number of points from geometry type %s" % (
                shape.geom_type))

        return vector

    # TODO: refactor to vectorize any number of wkts
    @staticmethod
    def vectorize_two_wkts(wkt1, wkt2, number_of_points, simplify=False):
        """
        Convert two wkt geometries to low-level feature engineered 2D array
        :param wkt1: the first geometry in wkt
        :param wkt2: the second geometry in wkt
        :param number_of_points: the size of the first dimension of the nested array: the maximum number of points in
                                 the two combined wkt points
        :param simplify: optional, selecting douglas-peucker reduction of points if wkt points exceeds max_points
        :return vectors: a 2d array of vectorized representations of the input points
        """
        shape1 = loads(wkt1)
        shape2 = loads(wkt2)
        vectors = np.zeros((number_of_points, GEO_VECTOR_LEN))

        if shape1.geom_type == 'Polygon' and shape2.geom_type == 'MultiPolygon':
            # The vectorized shape1 will have ... elements:
            # [2:10] boolean: one-hot geometry type encoding
            points1 = [xy for xy in shape1.exterior.coords]  # Ignore any inner linear rings!
            points2 = [xy for xy in shape2.geoms[0].exterior.coords]  # Ignore any inner linear rings!
            if len(shape2.geoms) > 1:
                raise ValueError("Don't know how to process multipart geometries with more than one part")

            vectors = GeoVectorizer.vectorize_points(points1, vectors, shape1.geom_type, offset=0,
                                                     simplify=simplify)
            vectors = GeoVectorizer.vectorize_points(points2, vectors, shape2.geoms[0].geom_type, offset=len(points1),
                                                     is_last=True, simplify=simplify)
        elif shape1.geom_type == 'Polygon' and shape2.geom_type == 'Polygon':
            # The vectorized shape1 will have ... elements:
            # [2:10] boolean: one-hot geometry type encoding
            points1 = [xy for xy in shape1.exterior.coords]  # Ignore any inner linear rings!
            points2 = [xy for xy in shape2.exterior.coords]  # Ignore any inner linear rings!
            vectors = GeoVectorizer.vectorize_points(points1, vectors, shape1.geom_type, offset=0,
                                                     simplify=simplify)
            vectors = GeoVectorizer.vectorize_points(points2, vectors, shape2.geom_type, offset=len(points1),
                                                     is_last=True, simplify=simplify)
        else:
            raise ValueError("Don't know how to process geometry combinations of type %s and %s" % (
                shape1.geom_type, shape2.geom_type))

        return vectors

    @staticmethod
    def vectorize_points(points, vector, geom_type, offset=0, is_last=False, simplify=False):
        """
        Fill an array of vectors out of an array of points from a geometry
        :param points: the array of input points
        :param vector: a 2D vector, to be filled in as output of the function
        :param geom_type: a geometry type string, one of GEOMETRY_TYPES
        :param offset: offset in the vector, to be used to fill in a second point in the vector.
        :param is_last: extra offset for the last point in a geometry, to indicate a full stop.
        :param simplify: selecting douglas-peucker reduction of points if wkt points exceeds max_points
        :return vectors: a filled-in nested array of vectors.
        """

        copy = vector.copy()

        if len(points) > copy.shape[0]:
            if not simplify:
                raise ValueError('The number of points in the geometry exceeds the max points in the output '
                                 'specification, but the reduce_points parameter was set to False. Please set the '
                                 'reduce_points parameter to True to reduce the number of points, or increase the max '
                                 'points.')
            else:
                points = GeoVectorizer.recursive_simplify(copy.shape[0], points)

        for point_index, point in enumerate(points):
            geom_type_one_hot = GEOMETRY_TYPES.index(geom_type) + GEOM_TYPE_INDEX  # offset from coordinate entries
            # [11:14] boolean: [render, end of first geometry, end of second geometry]

            position = point_index + offset
            copy[position][X_INDEX] = point[0]
            copy[position][Y_INDEX] = point[1]
            copy[position][geom_type_one_hot] = True

            if point_index == len(points) - 1:
                if is_last:
                    copy[position:, STOP_INDEX + 1] = True
                else:
                    copy[position, STOP_INDEX] = True
            else:
                copy[position, RENDER_INDEX] = True

        return copy

    @staticmethod
    def recursive_simplify(max_points, points):
        """
        Search algorithm for reducing the number of points of a geometry
        :param max_points:
        :param points:
        :return:
        """
        log_tolerance = -10  # Log scale
        tolerance = math.pow(10, log_tolerance)
        line = geometry.LineString(points).simplify(tolerance)
        while len(line.coords) > max_points:
            line = line.simplify(tolerance)
            log_tolerance += 0.5
            tolerance = math.pow(10, log_tolerance)
            line = line.simplify(tolerance)
        points = line.coords
        return points

    @staticmethod
    def decypher(vector):
        """
        Decyphers a encoded 2D coordinate and one-hot vector back to a wkt geometry -
            the inverse of the vectorize_wkt method
        :param vector: a length 13 vector with 2 coordinate positions, 8 geom type positions and 3 render positions
        :return: a string of a well-known text geometry
        """
        geom_type = GEOMETRY_TYPES[np.argmax(vector[0][GEOM_TYPE_INDEX:RENDER_INDEX])]
        wkt = geom_type.upper() + wkt_start[geom_type]

        # TODO: refactor to use actual filled geometry collections
        # If an empty geometrycollection:
        if geom_type == 'GeometryCollection':
            return wkt

        for point in vector:
            action_type = np.argmax(point[RENDER_INDEX:])

            wkt += ' '.join([str(point[0]), str(point[1])])
            if action_types[action_type] == 'render':
                wkt += ','
            elif action_types[action_type] == 'stop':
                wkt += wkt_end[geom_type] + '\n' + geom_type.upper() + wkt_start[geom_type]
            else:  # action_types[action_type] == 'full stop':
                break

        return wkt + wkt_end[geom_type]

    def decypher_gmm_geom(self, vector, sample_size=None, use_covariance=False):
        """
        Decyphers a encoded 2d vector of bivariate gaussian mixture model component(s) parameters and one-hot vectors
            back to a wkt geometry. Beware that you need to provide the number of gaussian mixture components in the
            class constructor.
        :param vector: rank 2 input vector of sequences with parameters for a gaussian mixture model, two one-hot
            vectors for geometry type and 'pen state' or stop indicator, all as a concatenated sequence
        :param sample_size: optional number of points to sample from each mixture component. Returns a MultiPoint geom.
        :param use_covariance: whether or not to apply the covariance matrix in sampling. Useful for monitoring the
                                distribution of the data, but can create a very cloudy result
        :return a shapely geometry of either:
                    if sample_size > 0: max_points * sample size sampled 2d MultiPoint geometry
                    if !sample_size: the geometry type and points indicated by the vector
        """
        one_hot_start = (self.gmm_size * 6)
        render_state_start = one_hot_start + GEOM_TYPE_LEN
        (max_points, _) = vector[:, 0:one_hot_start].shape

        render_states = np.argmax(vector[:, render_state_start:], axis=1).tolist()
        full_stop_index = render_states.index(2)  # first element with full stop bit
        geom_type = GEOMETRY_TYPES[int(
            np.median(
                np.argmax(
                    vector[:, one_hot_start:render_state_start], axis=1)[:full_stop_index]))]
        components = np.reshape(
            vector[:full_stop_index, 0:one_hot_start],  # use only the gaussian components
            (full_stop_index, int(one_hot_start / 6), 6))  # reshape to a rank 3 to split the different components
        most_likely_components = np.argmax(components[:, :, 5], axis=1)  # highest-scoring component

        if sample_size:  # the user wants a set of sampled points
            sample = []
            for index, highest in enumerate(most_likely_components):
                [mean_x, mean_y, sigma_x, sigma_y, rho] = components[index, highest, 0:5]
                [sigma_x, sigma_y] = np.abs([sigma_x, sigma_y])
                rho = np.tanh(rho)

                # use of the covariance matrix is standard off, since it tends to cloud the output too much
                covariance = [[sigma_x, rho], [rho, sigma_y]] if use_covariance else [[0, 0], [0, 0]]
                sampled = np.random.multivariate_normal(
                    mean=[mean_x, mean_y],
                    cov=covariance,
                    size=sample_size)
                # sample.append(np.repeat([mean_x, mean_y], 10))
                sample.append(sampled)

            sample = np.reshape(sample, ((index + 1) * sample_size, 2))  # reshape to 2d points
            return geometry.MultiPoint(sample)
        else:  # the user wants a geometry back
            if geom_type == 'Polygon':
                xy = [[component[likely, 0], component[likely, 1]]
                      for likely, component in zip(most_likely_components, components)]
                geom = geometry.Polygon(xy)
            else:
                raise ValueError('Decyphering geometries of type ' + geom_type + ' isn\'t supported yet')
            return geom
