from shapely.wkt import loads
import numpy as np

GEOMETRY_TYPES = ["Geometry", "Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon",
                  "GeometryCollection"]


class Vectorizer:
    @staticmethod
    def vectorize_wkt(wkt1, wkt2):
        shape1 = loads(wkt1)
        shape2 = loads(wkt2)

        if shape1.geom_type == 'Polygon' and shape2.geom_type == 'MultiPolygon':
            # The vectorized shape1 will have ... elements:
            # [2:10] boolean: one-hot geometry type encoding
            points1 = [xy for xy in shape1.exterior.coords]  # Ignore any inner linear rings!
            points2 = [xy for xy in shape2.geoms[0].exterior.coords]  # Ignore any inner linear rings!
            if len(shape2.geoms) > 1:
                raise ValueError("Don't know how to process multipart geometries with more than one part")

            vector_length = 13
            vectors = np.zeros((len(points1) + len(points2), vector_length * 2))
            vectors = Vectorizer.vectorize_points(points1, vectors, shape1.geom_type, offset=0)
            vectors = Vectorizer.vectorize_points(points2, vectors, shape2.geoms[0].geom_type, offset=vector_length,
                                                  is_last=True)
        elif shape1.geom_type == 'Polygon' and shape2.geom_type == 'Polygon':
            # The vectorized shape1 will have ... elements:
            # [2:10] boolean: one-hot geometry type encoding
            points1 = [xy for xy in shape1.exterior.coords]  # Ignore any inner linear rings!
            points2 = [xy for xy in shape2.exterior.coords]  # Ignore any inner linear rings!
            vector_length = 13
            vectors = np.zeros((len(points1) + len(points2), vector_length * 2))
            vectors = Vectorizer.vectorize_points(points1, vectors, shape1.geom_type, offset=0)
            vectors = Vectorizer.vectorize_points(points2, vectors, shape2.geom_type, offset=vector_length,
                                                  is_last=True)
        else:
            raise ValueError("Don't know how to process geometry combinations of type %s and %s" % (
                shape1.geom_type, shape2.geom_type))

        return vectors

    @staticmethod
    def vectorize_points(points, vectors, geom_type, offset=0, is_last=False):
        for index, point in enumerate(points):
            x_index = 0 + offset  # float: the X coordinate
            y_index = 1 + offset  # float: the Y coordinate
            stop_index = 11 + offset + is_last  # [11:13] boolean: [end of first geometry, end of second geometry]

            geom_type_one_hot = GEOMETRY_TYPES.index(geom_type) + 2  # offset from coordinate entries
            vectors[index][x_index] = point[0]
            vectors[index][y_index] = point[1]
            vectors[index][geom_type_one_hot] = True
            if index == len(points) - 1:
                vectors[index][stop_index] = True

        return vectors
