from shapely.wkt import loads
import numpy as np

GEOMETRY_TYPES = ["GeometryCollection", "Point", "LineString", "Polygon", "MultiPoint", "MultiLineString",
                  "MultiPolygon", "Geometry"]
X_INDEX = 0  # float: the X coordinate
Y_INDEX = 1  # float: the Y coordinate
GEOM_TYPE_INDEX = 5
GEOM_TYPE_LEN = 8
RENDER_INDEX = GEOM_TYPE_INDEX + GEOM_TYPE_LEN
STOP_INDEX = RENDER_INDEX + 1
FULL_STOP_INDEX = STOP_INDEX + 1
GEO_VECTOR_LEN = FULL_STOP_INDEX + 1  # The amount of positions needed to describe the features of a geometry point

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
    def __init__(self, gmm_size):
        self.gmm_size = gmm_size

    @staticmethod
    def max_points(wkt_set1, wkt_set2):
        max_points = 0

        for index, _ in enumerate(wkt_set1):
            number_of_points = GeoVectorizer.num_points_from_wkt(wkt_set1[index]) + \
                               GeoVectorizer.num_points_from_wkt(wkt_set2[index])
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
            if not interpolated[index, RENDER_INDEX]:
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
    def vectorize_wkt(wkt, number_of_points):
        shape = loads(wkt)
        vectors = np.zeros((number_of_points, GEO_VECTOR_LEN))

        if shape.geom_type == 'Polygon':
            vectors = GeoVectorizer.vectorize_points(shape.exterior.coords, vectors,
                                                     shape.geom_type, is_last=True)
        elif shape.geom_type == 'MultiPolygon':
            if len(shape.geoms) > 1:
                raise ValueError("Don't know how to process multipart geometries with more than one part")
            vectors = GeoVectorizer.vectorize_points(shape.geom[0].exterior.coords, vectors,
                                                     shape.geom_type, is_last=True)
        elif shape.geom_type == 'GeometryCollection':
            if not len(shape.geoms) == 0:  # not GEOMETRYCOLLECTION EMPTY
                raise ValueError("Don't know how to process non-empty GeometryCollection type")
        elif shape.geom_type == 'Point':
            vectors = GeoVectorizer.vectorize_points(shape.coords, vectors,
                                                     shape.geom_type, is_last=True)
        else:
            raise ValueError("Don't know how to get the number of points from geometry type %s" % (
                shape.geom_type))

        return vectors

    @staticmethod
    def vectorize_two_wkts(wkt1, wkt2, number_of_points):
        """
        Convert two wkt geometries to low-level feature engineered 2D array
        :param wkt1: the first geometry in wkt
        :param wkt2: the second geometry in wkt
        :param number_of_points: the size of the first dimension of the nested array: the maximum number of points in
                                 the two combined wkt points
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

            vectors = GeoVectorizer.vectorize_points(points1, vectors, shape1.geom_type, offset=0)
            vectors = GeoVectorizer.vectorize_points(points2, vectors, shape2.geoms[0].geom_type, offset=len(points1),
                                                     is_last=True)
        elif shape1.geom_type == 'Polygon' and shape2.geom_type == 'Polygon':
            # The vectorized shape1 will have ... elements:
            # [2:10] boolean: one-hot geometry type encoding
            points1 = [xy for xy in shape1.exterior.coords]  # Ignore any inner linear rings!
            points2 = [xy for xy in shape2.exterior.coords]  # Ignore any inner linear rings!
            vectors = GeoVectorizer.vectorize_points(points1, vectors, shape1.geom_type, offset=0)
            vectors = GeoVectorizer.vectorize_points(points2, vectors, shape2.geom_type, offset=len(points1),
                                                     is_last=True)
        else:
            raise ValueError("Don't know how to process geometry combinations of type %s and %s" % (
                shape1.geom_type, shape2.geom_type))

        return vectors

    @staticmethod
    def vectorize_points(points, vectors, geom_type, offset=0, is_last=False):
        """
        Fill an array of ML-vectors out of an array of points from a geometry to be used in a recurrent neural net.
        :param points: the array of input points
        :param vectors: a nested array of vectors, to be filled in as output of the function
        :param geom_type: a geometry type string, one of GEOMETRY_TYPES
        :param offset: offset in the vector, to be used to fill in a second point in the vector.
        :param is_last: extra offset for the last point in a geometry, to indicate a full stop.
        :return vectors: a filled-in nested array of vectors.
        """
        for point_index, point in enumerate(points):
            geom_type_one_hot = GEOMETRY_TYPES.index(geom_type) + GEOM_TYPE_INDEX  # offset from coordinate entries
            # [11:14] boolean: [render, end of first geometry, end of second geometry]

            vectors[point_index + offset][X_INDEX] = point[0]
            vectors[point_index + offset][Y_INDEX] = point[1]
            vectors[point_index + offset][geom_type_one_hot] = True

            if point_index == len(points) - 1:
                vectors[point_index + offset][STOP_INDEX + is_last] = True
            else:
                vectors[point_index + offset][RENDER_INDEX] = True

        return vectors

    @staticmethod
    def decypher(vector):
        """
        Decyphers a encoded 2D coordinate and one-hot vector back to a wkt geometry
        :param vector:
        :return: a \n delimited string of one or more well-known text geometries
        """
        geom_type = GEOMETRY_TYPES[np.argmax(vector[0][GEOM_TYPE_INDEX:RENDER_INDEX])]
        wkt = geom_type.upper() + wkt_start[geom_type]

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

    def decypher_gmm_geom(self, vector, sample_size=10):
        """
        Decyphers a encoded vector of 2D gaussian mixture model components and one-hot vectors back to a wkt geometry
        :param vector:  a rank 2 input vector of sequences from a gaussian mixture model and two one-hot vectors for
                        geometry type and 'pen state' or stop indicator, all as a concatenated sequence
        :param sample_size: number of points to sample from each mixture component
        :return a list of max_points * sample size sampled 2d points
        """
        one_hot_start = (self.gmm_size * 6)
        render_state_start = one_hot_start + GEOM_TYPE_LEN
        (max_points, _) = vector[:, 0:one_hot_start].shape
        components = np.reshape(
            vector[:, 0:one_hot_start],  # use only the gaussian components
            (max_points, int(one_hot_start / 6), 6))   # reshape to a rank 3 to split the different components

        most_likely = np.argmax(components[:, :, 5], axis=1)  # the highest-scoring component

        sample = []
        for index, highest in enumerate(most_likely):
            render_state = np.argmax(vector[index, render_state_start:])
            if action_types[render_state] == 'full stop':
                print('Stop on node', index)
                index -= 1  # Prevent errors on reshaping in case of no full stop encounter
                break

            [mean_x, mean_y, sigma_x, sigma_y, rho] = components[index, highest, 0:5]
            sampled = np.random.multivariate_normal(
                mean=[mean_x, mean_y],
                cov=[[0, 0], [0, 0]],
                size=sample_size)
            # sample.append(np.repeat([mean_x, mean_y], 10))
            sample.append(sampled)

        return np.reshape(sample, ((index + 1) * sample_size, 2))  # reshape to 2d points
