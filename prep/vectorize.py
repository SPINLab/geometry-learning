import numpy as np
import pandas

from model.topoml_util.GeoVectorizer import GeoVectorizer, GEO_VECTOR_LEN, RENDER_INDEX, FULL_STOP_INDEX, STOP_INDEX

TOPOLOGY_TRAINING_CSV = '../files/topology-training.csv'
GEODATA_VECTORIZED = '../files/geodata_vectorized.npz'
MAX_SEQUENCE_LEN = 300

print('Reading data...')
training_data = pandas.read_csv(TOPOLOGY_TRAINING_CSV)
truncated_data = np.array([
    record for record in training_data.values
    if len(record[0] + ';' + record[1]) <= MAX_SEQUENCE_LEN])
raw_training_set = training_data['brt_wkt'] + ';' + training_data['osm_wkt']
raw_target_set = training_data['intersection_wkt']
print(len(raw_training_set), 'data points in training set')

brt_wkt = truncated_data[:, 0]
osm_wkt = truncated_data[:, 1]
intersection_set = truncated_data[:, 2]
centroid_distance = truncated_data[:, 3]
geom_distance = truncated_data[:, 4]
brt_centroid = [GeoVectorizer.vectorize_wkt(point, 1) for point in truncated_data[:, 5]]
osm_centroid = [GeoVectorizer.vectorize_wkt(point, 1) for point in truncated_data[:, 6]]
brt_centroid_rd = [GeoVectorizer.vectorize_wkt(point, 1) for point in truncated_data[:, 7]]
osm_centroid_rd = [GeoVectorizer.vectorize_wkt(point, 1) for point in truncated_data[:, 8]]
intersection_surface = truncated_data[:, 9]

training_set = brt_wkt + ';' + osm_wkt
print(len(training_set), 'max length data points in training set')

print('Vectorizing WKT geometries...')
max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
training_vectors = np.zeros((len(training_set), max_points, GEO_VECTOR_LEN))
intersection_vectors = np.zeros((len(intersection_set), max_points, GEO_VECTOR_LEN))
brt_max_points = GeoVectorizer.max_points(brt_wkt)
brt_vectors = np.zeros((len(brt_wkt), brt_max_points, GEO_VECTOR_LEN))
osm_max_points = GeoVectorizer.max_points(osm_wkt)
osm_vectors = np.zeros((len(osm_wkt), osm_max_points, GEO_VECTOR_LEN))

broken_records = []
for record_index in range(len(brt_wkt)):
    try:
        training_vector = GeoVectorizer.vectorize_two_wkts(brt_wkt[record_index], osm_wkt[record_index], max_points)
        target_vector = GeoVectorizer.vectorize_wkt(intersection_set[record_index], max_points)
        brt_vector = GeoVectorizer.vectorize_wkt(brt_wkt[record_index], brt_max_points)
        osm_vector = GeoVectorizer.vectorize_wkt(osm_wkt[record_index], osm_max_points)
    except Exception as e:
        print('Creating dummy record', record_index, ':', e)
        training_vectors[record_index, 0:, FULL_STOP_INDEX] = 1
        intersection_vectors[record_index, 0:, FULL_STOP_INDEX] = 1
        brt_vectors[record_index, 0:, FULL_STOP_INDEX] = 1
        osm_vectors[record_index, 0:, FULL_STOP_INDEX] = 1

        broken_records.append(record_index)
        continue

    # Fill in the numpy zeros tensor value by value
    for point_index, point in enumerate(training_vector):
        for feature_index, value in enumerate(point):
            training_vectors[record_index, point_index, feature_index] = value

    for point_index, point in enumerate(target_vector):
        for feature_index, value in enumerate(point):
            intersection_vectors[record_index, point_index, feature_index] = value

    for point_index, point in enumerate(brt_vector):
        for feature_index, value in enumerate(point):
            brt_vectors[record_index, point_index, feature_index] = value

    for point_index, point in enumerate(osm_vector):
        for feature_index, value in enumerate(point):
            osm_vectors[record_index, point_index, feature_index] = value

# Concatenate centroids
centroids = np.append(brt_centroid, osm_centroid, axis=1)
# Fix the stop and full stop bits on the first centroid
centroids[:, 0, FULL_STOP_INDEX] = 0
centroids[:, 0, STOP_INDEX] = 1

centroids_rd = np.append(brt_centroid_rd, osm_centroid_rd, axis=1)
# Fix the stop and full stop bits on the first centroid
centroids_rd[:, 0, FULL_STOP_INDEX] = 0
centroids_rd[:, 0, STOP_INDEX] = 1

# Make room for extra gaussian parameters in distance and surface properties
centroid_distance = np.reshape(centroid_distance, (len(centroid_distance), 1, 1))
centroid_distance = np.insert(centroid_distance, 1, 0, axis=2)
geom_distance = np.reshape(geom_distance, (len(geom_distance), 1, 1))
geom_distance = np.insert(geom_distance, 1, 0, axis=2)
intersection_surface = np.reshape(intersection_surface, (len(intersection_surface), 1, 1))
intersection_surface = np.insert(intersection_surface, 1, 0, axis=2)

print('Saving compressed numpy data file', GEODATA_VECTORIZED)

np.savez_compressed(
    GEODATA_VECTORIZED,
    input_geoms=training_vectors,               # Two concatenated vectorized geoms in WGS84 lon/lat, with 25% overlap
    brt_vectors=brt_vectors,                    # BRT building vector in WGS84 lon/lat
    osm_vectors=osm_vectors,                    # OSM building vector in WGS84 lon/lat
    intersection=intersection_vectors,          # Vectorized intersection geoms in WGS84 lon/lat
    centroid_distance=centroid_distance,        # Distance in meters between the centroids
    geom_distance=geom_distance,                # Distance in meters between the geometries, 0 if intersecting
    brt_centroid=brt_centroid,                  # Centroid point in WGS84 lon/lat of the BRT geometry
    osm_centroid=osm_centroid,                  # Centroid point in WGS84 lon/lat of the OSM geometry
    centroids=centroids,                        # Two centroid points for BRT and OSM in WGS84 lon/lat
    centroids_rd=centroids_rd,                  # Two centroid points for BRT and OSM in Netherlands RD meters
    intersection_surface=intersection_surface,  # Surface in square meters of the intersection_surface_area
)
print('Saved vectorized geometries to', GEODATA_VECTORIZED)
