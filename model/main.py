import pandas
import numpy as np
from model.topoml_util.Tokenizer import Tokenize
from model.topoml_util.Vectorizer import Vectorizer

TOPOLOGY_TRAINING_CSV = '../files/topology-training.csv'
MAX_SEQUENCE_LEN = 220
TRAIN_VALIDATE_SPLIT = 0.1
EPOCHS = 30

print('Reading data...')
training_data = pandas.read_csv(TOPOLOGY_TRAINING_CSV)
raw_training_set = training_data['brt_wkt'] + ';' + training_data['osm_wkt']
raw_target_set = training_data['intersection_wkt']
print(len(raw_training_set), 'data points in training set')

(training_set, target_set) = Tokenize.truncate(MAX_SEQUENCE_LEN,
                                               raw_training_set,
                                               raw_target_set)

print(len(target_set), 'max length data points in training set')

brt_wkt = []
osm_wkt = []
for record in training_set:
    sets = record.split(';')
    brt_wkt.append(sets[0])
    osm_wkt.append(sets[1])

vectorized = []
for index in range(len(brt_wkt)):
    vectorized.append(Vectorizer.vectorize_wkt(brt_wkt[index], osm_wkt[index]))
data_points = len(vectorized)
(max_points, features) = max([array.shape for array in vectorized])
np_array = np.zeros((data_points, max_points, features))
for record_index, record in enumerate(vectorized):
    for point_index, point in enumerate(record):
        np_array[record_index][point_index] = point

print('Tokenizing WKT sequences...')
