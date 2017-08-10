import numpy as np
import pandas

from topoml_util.GeoVectorizer import GeoVectorizer, GEO_VECTOR_LEN


def truncate(max_len, untruncated_training_set, untruncated_target_set):
    """
    Method for truncating the training and target set to fit the maximum
        sequence length, batch and validation set size
    :param max_len: maximum length of characters per sequence/sentence
    :param untruncated_training_set: untruncated list of input sequences
    :param untruncated_target_set: untruncated list of target output sequences
    :return: training_set, target_set: a tuple of truncated training and target sets
    """
    training_set = []
    target_set = []

    # Restrict input to be of less or equal length as the maximum length.
    for index, record in enumerate(untruncated_training_set):
        if len(record) <= max_len:
            training_set.append(record)
            target_set.append(untruncated_target_set[index])

    return training_set, target_set

TOPOLOGY_TRAINING_CSV = '../files/topology-training.csv'
GEODATA_VECTORIZED = '../files/geodata_vectorized.npz'
MAX_SEQUENCE_LEN = 220

print('Reading data...')
training_data = pandas.read_csv(TOPOLOGY_TRAINING_CSV)
raw_training_set = training_data['brt_wkt'] + ';' + training_data['osm_wkt']
raw_target_set = training_data['intersection_wkt']
print(len(raw_training_set), 'data points in training set')

(training_set, target_set) = truncate(MAX_SEQUENCE_LEN, raw_training_set, raw_target_set)
print(len(target_set), 'max length data points in training set')

brt_wkt = []
osm_wkt = []
for record in training_set:
    sets = record.split(';')
    brt_wkt.append(sets[0])
    osm_wkt.append(sets[1])
max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)

print('Vectorizing WKT geometries...')
training_vectors = np.zeros((len(target_set), max_points, GEO_VECTOR_LEN))
target_vectors = np.zeros((len(target_set), max_points, GEO_VECTOR_LEN))

for record_index in range(len(brt_wkt)):
    training_vector = GeoVectorizer.vectorize_two_wkts(brt_wkt[record_index], osm_wkt[record_index], max_points)
    for point_index, point in enumerate(training_vector):
        for feature_index, feature in enumerate(point):
            training_vectors[record_index][point_index][feature_index] = feature

    target_vector = GeoVectorizer.vectorize_wkt(training_set[record_index], max_points)
    for point_index, point in enumerate(target_vector):
        for feature_index, feature in enumerate(point):
            target_vectors[record_index][point_index][feature_index] = feature

np.savez_compressed(GEODATA_VECTORIZED, X=training_vectors, y=target_vectors)
print('Saved vectorized geometries to %s' % GEODATA_VECTORIZED)
