import os

import collections
from model.topoml_util.geom_fourier_descriptors import geom_fourier_descriptors
from model.topoml_util.GeoVectorizer import GeoVectorizer
from pandas import read_csv
from shapely import wkt
import numpy as np

SOURCE = '../files/archaeology/combined_arch_feat.csv'
TRAIN_DATA_FILE = '../files/archaeology/archeo_features_train.npz'
TEST_DATA_FILE = '../files/archaeology/archeo_features_test.npz'
SANE_NUMBER_OF_POINTS = 256
TRAIN_TEST_SPLIT = 0.1
FOURIER_DESCRIPTOR_ORDER = 16  # The axis 0 size
NUMBER_OF_FILES = 4
MINIMUM_CLASS_OCCURRENCE = 1000

if not os.path.isfile(SOURCE):
    raise FileNotFoundError('Unable to locate %s. Please run the prep/get-data.sh script first' % SOURCE)

print('Preprocessing archaeological features...')
with open(SOURCE) as file:
    df = read_csv(file)

aardspoor__as_matrix = df['Aardspoor'].as_matrix()
wkt__as_matrix = df['WKT'].as_matrix()

class_count = dict(collections.Counter([f_type for f_type in aardspoor__as_matrix if type(f_type) == str]))
# print('class count:', class_count)
included_classes = [f_type for f_type, count in class_count.items() if count > MINIMUM_CLASS_OCCURRENCE]
print('Included classes:', included_classes)

# geometry vectors
print('Creating geometry vectors...')
feature_types = []
wkts = []
wkt_vectors = []
for index, (feature, geom) in enumerate(zip(aardspoor__as_matrix, wkt__as_matrix)):
    if feature in included_classes:
        try:
            wkt_vectors.append(
                GeoVectorizer.vectorize_wkt(geom, SANE_NUMBER_OF_POINTS, simplify=True))
            wkts.append(geom)
        except Exception as e:
            print('skipping record on account of geometry entry in {0} on line {1} with error: {2}'.format(
                SOURCE, index + 2, e))
            continue
        feature_types.append(feature)
        wkts.append(geom)

class_distr = collections.Counter(feature_types)
print('data points:', len(feature_types), 'over', len(dict(class_distr)), 'classes')
print('class distribution:', class_distr)

# Convert types to indices
feature_types = [included_classes.index(f_type) for f_type in feature_types]

print('Creating geometry fourier descriptors...')
shapes = []
for wkt_string in wkts:  # create the descriptors on the untruncated geoms
    shape = wkt.loads(wkt_string)
    # If multipart multipolygon: select the largest, but it will throw off the accuracy a bit.
    if shape.geom_type == 'MultiPolygon':
        if len(shape.geoms) > 1:
            geometries = sorted(shape.geoms, key=lambda x: x.area)
            shapes.append(geometries[-1])
        else:
            shapes.append(shape.geoms[0])
    else:
        shapes.append(shape)

fourier_descriptors = geom_fourier_descriptors(shapes, FOURIER_DESCRIPTOR_ORDER)

# Split and save data
train_test_split_index = round(TRAIN_TEST_SPLIT * len(wkts))

training_data = {
    'geoms': wkt_vectors[:-train_test_split_index],
    'fourier_descriptors': fourier_descriptors[:-train_test_split_index],
    'feature_type': feature_types[:-train_test_split_index],
    'feature_type_index': included_classes,
}

test_data = {
    'geoms': wkt_vectors[-train_test_split_index:],
    'fourier_descriptors': fourier_descriptors[-train_test_split_index:],
    'feature_type': feature_types[-train_test_split_index:],
    'feature_type_index': included_classes,
}

print('Saving training and test data files...')

np.savez_compressed(
    TRAIN_DATA_FILE,
    geoms=training_data['geoms'],
    fourier_descriptors=training_data['fourier_descriptors'],
    feature_type=training_data['feature_type'],
)

# Test data is small enough to put in one archive
np.savez_compressed(
    TEST_DATA_FILE,
    geoms=test_data['geoms'],
    fourier_descriptors=test_data['fourier_descriptors'],
    feature_type=test_data['feature_type'],
)

print('Done!')
