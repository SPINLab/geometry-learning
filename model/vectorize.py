import numpy as np
import pandas

from topoml_util.GeoVectorizer import GeoVectorizer, GEO_VECTOR_LEN

TOPOLOGY_TRAINING_CSV = '../files/topology-training.csv'
GEODATA_VECTORIZED = '../files/geodata_vectorized.npz'
MAX_SEQUENCE_LEN = 220

print('Reading data...')
training_data = pandas.read_csv(TOPOLOGY_TRAINING_CSV)
truncated_data = \
    np.array([record for record in training_data.values if len(record[0] + ';' + record[1]) <= MAX_SEQUENCE_LEN])
raw_training_set = training_data['brt_wkt'] + ';' + training_data['osm_wkt']
raw_target_set = training_data['intersection_wkt']
print(len(raw_training_set), 'data points in training set')

brt_wkt = truncated_data[:, 0]
osm_wkt = truncated_data[:, 1]
intersection_set = truncated_data[:, 2]
centroid_distance = truncated_data[:, 3]
geom_distance = truncated_data[:, 4]

training_set = brt_wkt + ';' + osm_wkt
print(len(training_set), 'max length data points in training set')

print('Vectorizing WKT geometries...')
max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)
training_vectors = np.zeros((len(training_set), max_points, GEO_VECTOR_LEN))
intersection_vectors = np.zeros((len(intersection_set), max_points, GEO_VECTOR_LEN))

for record_index in range(len(brt_wkt)):
    training_vector = GeoVectorizer.vectorize_two_wkts(brt_wkt[record_index], osm_wkt[record_index], max_points)
    for point_index, point in enumerate(training_vector):
        for feature_index, feature in enumerate(point):
            training_vectors[record_index][point_index][feature_index] = feature

    target_vector = GeoVectorizer.vectorize_wkt(training_set[record_index], max_points)
    for point_index, point in enumerate(target_vector):
        for feature_index, feature in enumerate(point):
            intersection_vectors[record_index][point_index][feature_index] = feature

# Make room for extra gaussian parameters
centroid_distance = np.reshape(centroid_distance, (len(centroid_distance), 1, 1))
centroid_distance = np.insert(centroid_distance, 1, 0, axis=2)
geom_distance = np.reshape(geom_distance, (len(geom_distance), 1, 1))
geom_distance = np.insert(geom_distance, 1, 0, axis=2)

np.savez_compressed(GEODATA_VECTORIZED,
                    input_geoms=training_vectors,
                    intersection=intersection_vectors,
                    centroid_distance=centroid_distance,
                    geom_distance=geom_distance,
                    )
print('Saved vectorized geometries to %s' % GEODATA_VECTORIZED)
