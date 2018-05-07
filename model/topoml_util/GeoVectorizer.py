import re

from shapely.wkt import loads
import numpy as np
import math

# TODO: refactor GEOMETRY_TYPES to use shapely.geometry.base.GEOMETRY_TYPE
GEOMETRY_TYPES = ["GeometryCollection", "Point", "LineString", "Polygon", "MultiPoint", "MultiLineString",
                  "MultiPolygon", "Geometry"]
X_INDEX = 0  # the X coordinate position
Y_INDEX = 1  # the Y coordinate position
GEOM_TYPE_INDEX = Y_INDEX + 1  # Start index of the geometry type
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
    def num_points_from_wkt(wkt):
        # A 2D point in WKT is a set of two numerical values, separated by a space:
        # marked by two decimal values on either side
        shape = loads(wkt)
        pattern = '\d \d'

        if shape.has_z:
            pattern += ' \d'

        number_of_points = len(re.findall(pattern, shape.wkt))
        return number_of_points

    @staticmethod
    def vectorize_wkt(wkt, max_points, simplify=False, fixed_size=False):
        """
        Convert wkt geometry to a numpy array of real values. The size of the vector is equal to:
            if fixed_size=False: p where p is the size of the set of points in the geometry;
            is fixed_size=True: max_points, padded with zeros.
        :param wkt: the geometry as wkt string
        :param max_points: the maximum size of the first output dimension: the maximum number of points
        :param simplify: optional, selecting reduction of points if wkt points exceeds max_points
        :param fixed_size: If set to True, the function returns a matrix of size max_points
        :return vectors: a 2d numpy array as vectorized representation of the input geometry
        """
        shape = loads(wkt)
        total_points = GeoVectorizer.num_points_from_wkt(shape.wkt)  # use the shapely wkt form for consistency

        if total_points > max_points:
            if not simplify:
                raise ValueError('The number of points in the geometry exceeds the max_points but the reduce_points '
                                 'parameter was set to False. Please set the reduce_points parameter to True to reduce '
                                 'the number of points, or increase max_points parameter.')
            else:
                shape = GeoVectorizer.recursive_simplify(max_points, shape)
                total_points = GeoVectorizer.num_points_from_wkt(shape.wkt)

        if shape.geom_type == 'Polygon':
            geom_matrix = GeoVectorizer._vectorize_polygon(shape)
        elif shape.geom_type == 'MultiPolygon':
            # noinspection PyUnresolvedReferences
            geom_matrix = np.concatenate(
                [GeoVectorizer._vectorize_polygon(geom) for geom in shape.geoms], axis=0)
            geom_matrix[total_points - 1, STOP_INDEX] = 0
            # noinspection PyUnresolvedReferences
            geom_matrix = np.append(geom_matrix, np.zeros((max_points - total_points, GEO_VECTOR_LEN)), axis=0)
            geom_matrix[total_points - 1:, FULL_STOP_INDEX] = 1  # Manually set full stop bits
        elif shape.geom_type == 'GeometryCollection':
            if len(shape.geoms) > 0:  # not GEOMETRYCOLLECTION EMPTY
                raise ValueError("Don't know how to process non-empty GeometryCollection type")
            # noinspection PyUnresolvedReferences
            geom_matrix = np.zeros((1, GEO_VECTOR_LEN))
            geom_matrix[:, FULL_STOP_INDEX] = 1  # Manually set full stop bits
        elif shape.geom_type == 'Point':
            geom_matrix = GeoVectorizer._vectorize_points(shape.coords, shape.geom_type, is_last=True)
        else:
            raise ValueError("Don't know how to get the number of points from geometry type {}".format(shape.geom_type))

        if fixed_size:
            pad_shape = ((0, max_points - len(geom_matrix)), (0, 0))
            geom_matrix = np.pad(geom_matrix, pad_shape, mode='constant')
        return geom_matrix

    @staticmethod
    def _vectorize_polygon(shape):
        return GeoVectorizer._vectorize_points(shape.exterior.coords, shape.geom_type, is_last=True)

    @staticmethod
    def _vectorize_points(points, geom_type, is_last=False):
        """
        Fill an array of vectors out of an array of points from a geometry
        :param points: the array of input points
        :param geom_type: a geometry type string, one of GEOMETRY_TYPES
        :param is_last: extra offset for the last point in a geometry, to indicate a full stop.
        :return matrix: a matrix representation of the points.
        """
        # noinspection PyUnresolvedReferences
        matrix = np.zeros((len(points), GEO_VECTOR_LEN))

        for point_index, point in enumerate(points):
            geom_type_one_hot = GEOMETRY_TYPES.index(geom_type) + GEOM_TYPE_INDEX  # offset from coordinate entries
            # [11:14] boolean: [render, end of first geometry, end of second geometry]

            matrix[point_index, X_INDEX] = point[0]
            matrix[point_index, Y_INDEX] = point[1]
            matrix[point_index, geom_type_one_hot] = True

            if point_index == len(points) - 1:
                if is_last:
                    matrix[point_index, FULL_STOP_INDEX] = True
                else:
                    matrix[point_index, STOP_INDEX] = True
            else:
                matrix[point_index, RENDER_INDEX] = True

        return matrix

    @staticmethod
    def recursive_simplify(max_points, shape):
        """
        Search algorithm for reducing the number of points of a geometry
        :param max_points:
        :param shape: A shapely shape
        :return:
        """
        log_tolerance = -10  # Log scale
        tolerance = math.pow(10, log_tolerance)
        shape = shape.simplify(tolerance)
        while len(re.findall('\d \d', shape.wkt)) > max_points:
            log_tolerance += 0.5
            tolerance = math.pow(10, log_tolerance)
            shape = shape.simplify(tolerance)
        return shape
