import collections
import os
import sys

import numpy as np
from pandas import read_csv
from shapely import wkt

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from model.topoml_util.GeoVectorizer import GeoVectorizer
from model.topoml_util.geom_fourier_descriptors import create_geom_fourier_descriptor

SOURCE = '../files/archaeology/combined_arch_feat.csv'
TRAIN_DATA_FILE = '../files/archaeology/archaeo_features_train.npz'
TEST_DATA_FILE = '../files/archaeology/archaeo_features_test.npz'
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
print('Creating geometry vectors and descriptors...')
feature_types = []
shapes = []  # shapely shapes
wkt_vectors = []
fourier_descriptors = []
errors = 0

for index, (feature, geom) in enumerate(zip(aardspoor__as_matrix, wkt__as_matrix)):
    if feature in included_classes:
        try:
            wkt_vector = GeoVectorizer.vectorize_wkt(geom, SANE_NUMBER_OF_POINTS, simplify=True)

            # create the descriptors on the untruncated geoms
            shape = wkt.loads(geom)
            # If multipart multipolygon: reduce to polygon by selecting the largest,
            # but it will throw off the accuracy a bit.
            if shape.geom_type == 'MultiPolygon':
                if len(shape.geoms) > 1:
                    geometries = sorted(shape.geoms, key=lambda x: x.area)
                    shape = geometries[-1]
                else:
                    shape = shape.geoms[0]
                fourier_descriptor = create_geom_fourier_descriptor(shape, FOURIER_DESCRIPTOR_ORDER)
            elif shape.geom_type == 'Polygon':
                fourier_descriptor = create_geom_fourier_descriptor(shape, FOURIER_DESCRIPTOR_ORDER)
            else:
                print('skipping record: no (multi)polygon entry in {0} on line {1}'.format(
                    SOURCE, index + 2))
                errors += 1
                continue

        except Exception as e:
            print('skipping record on account of geometry entry in {0} on line {1} with error: {2}'.format(
                SOURCE, index + 2, e))
            errors += 1
            continue

        # Append the converted values if all went well
        wkt_vectors.append(wkt_vector)
        fourier_descriptors.append(fourier_descriptor)
        # Convert types to numerical indices
        feature_types.append(included_classes.index(feature))

print('created {0} data points with {1} errors'.format(len(wkt_vectors), errors))

class_distr = collections.Counter(feature_types)
print('data points:', len(feature_types), 'over', len(dict(class_distr)), 'classes')
print('class distribution:', class_distr)

# Split and save data
n_parts = int(1 / TRAIN_TEST_SPLIT)

wkt_vectors_parts = []
fourier_descriptors_parts = []
feature_types_parts = []

training_data = {
    'geoms': [],
    'fourier_descriptors': [],
    'feature_type': [],
    'feature_type_index': included_classes,
}

test_data = {
    'geoms': [],
    'fourier_descriptors': [],
    'feature_type': [],
    'feature_type_index': included_classes,
}

for part in range(n_parts):
    if part == 0:
        test_data['geoms'] = wkt_vectors[part::n_parts]
        test_data['fourier_descriptors'] = fourier_descriptors[part::n_parts]
        test_data['feature_type'] = feature_types[part::n_parts]
    elif part == 1:  # fill first part
        training_data['geoms'] = wkt_vectors[part::n_parts]
        training_data['fourier_descriptors'] = fourier_descriptors[part::n_parts]
        training_data['feature_type'] = feature_types[part::n_parts]
    else:  # append the rest
        training_data['geoms'] = np.append(training_data['geoms'], wkt_vectors[part::n_parts], axis=0)
        training_data['fourier_descriptors'] = np.append(training_data['fourier_descriptors'], fourier_descriptors[part::n_parts], axis=0)
        training_data['feature_type'] = np.append(training_data['feature_type'], feature_types[part::n_parts], axis=0)

print('Saving training and test data files...')

np.savez_compressed(
    TRAIN_DATA_FILE,
    geoms=training_data['geoms'],
    fourier_descriptors=training_data['fourier_descriptors'],
    feature_type=training_data['feature_type'],
    feature_type_index=training_data['feature_type_index'],
)

# Test data is small enough to put in one archive
np.savez_compressed(
    TEST_DATA_FILE,
    geoms=test_data['geoms'],
    fourier_descriptors=test_data['fourier_descriptors'],
    feature_type=test_data['feature_type'],
    feature_type_index=test_data['feature_type_index'],
)

print('Done!')
