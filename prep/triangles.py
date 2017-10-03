import numpy as np
from model.topoml_util.GeoVectorizer import GeoVectorizer
from shapely.geometry import Polygon
from shapely.wkt import loads

SET_SIZE = 100000
TRIANGLES = '../files/triangles.npz'


print('Creating triangles')
raw_training_vectors = np.random.normal(size=(SET_SIZE, 6, 2))
triangle_sets = np.array([[Polygon(point_set[0:3]).wkt, Polygon(point_set[3:]).wkt]
                          for point_set in raw_training_vectors])
max_points = GeoVectorizer.max_points(triangle_sets[:, 0], triangle_sets[:, 1])

print('Intersecting triangles and pruning')
intersection_area = []
intersection_vectors = []
for index, (a, b) in enumerate(triangle_sets):
    # if loads(a).intersection_surface_area(loads(b)).type == 'Polygon':  # constrain to actually intersecting
    intersection = loads(a).intersection(loads(b))
    intersection_area.append(intersection.area)
    intersection_vectors.append(GeoVectorizer.vectorize_wkt(intersection.wkt, 12))

training_vectors = np.reshape(raw_training_vectors, (SET_SIZE, 12))
(_, GEO_VECTOR_LEN) = np.array(training_vectors).shape
intersection_area = np.array(intersection_area)

print('Saving compressed numpy data file', TRIANGLES)

np.savez_compressed(
    TRIANGLES,
    point_sequence=training_vectors,          # Sets of two geometries in WGS84 lon/lat, 25% of them overlapping
    intersection_geoms=intersection_vectors,  # Geometries representing the intersection_surface_area in WGS84 lon/lat
    intersection_surface=intersection_area,   # Surface in square meters of the intersection_surface_area
)
print('Saved vectorized geometries to', TRIANGLES)
