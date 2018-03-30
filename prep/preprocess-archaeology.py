import os
from datetime import timedelta
from time import time
from zipfile import ZipFile

import collections
import numpy as np
from pandas import read_csv
from shapely import wkt

from model.topoml_util.GeoVectorizer import GeoVectorizer
from model.topoml_util.geom_fourier_descriptors import create_geom_fourier_descriptor
from prep.ProgressBar import ProgressBar

SOURCE_ZIP = '../files/archaeology/archaeology.csv.zip'
SOURCE_CSV = 'archaeology.csv'
TRAIN_DATA_FILE = '../files/archaeology/archaeology_order_30_train-'
TEST_DATA_FILE = '../files/archaeology/archaeology_order_30_test.npz'
SANE_NUMBER_OF_POINTS = 256
TRAIN_TEST_SPLIT = 0.1
FOURIER_DESCRIPTOR_ORDER = 30  # The axis 0 size
MINIMUM_CLASS_OCCURRENCE = 1000
NUMBER_OF_FILES = 3
SCRIPT_START = time()

if not os.path.isfile(SOURCE_ZIP):
    raise FileNotFoundError('Unable to locate %s. Please run the prep/get-data.sh script first' % SOURCE_ZIP)

print('Preprocessing archaeological features...')
zfile = ZipFile(SOURCE_ZIP)
df = read_csv(zfile.open(SOURCE_CSV))

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

pgb = ProgressBar()

for index, (feature, geom) in enumerate(zip(aardspoor__as_matrix, wkt__as_matrix)):
    pgb.update_progress(index/len(aardspoor__as_matrix))
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
                    SOURCE_CSV, index + 2))
                errors += 1
                continue

        except Exception as e:
            print('skipping record on account of geometry entry in {0} on line {1} with error: {2}'.format(
                SOURCE_CSV, index + 2, e))
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
# Create nicely even-sized chunks
for offset in range(NUMBER_OF_FILES):
    stride = NUMBER_OF_FILES  # just an alias

    part_geoms = training_data['geoms'][offset::stride]
    part_fourier_descriptors = training_data['fourier_descriptors'][offset::stride]
    part_feature_type = training_data['feature_type'][offset::stride]

    np.savez_compressed(
        TRAIN_DATA_FILE + str(offset),
        geoms=part_geoms,
        fourier_descriptors=part_fourier_descriptors,
        feature_type=part_feature_type,
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

runtime = time() - SCRIPT_START
print('Done in {}'.format(timedelta(seconds=runtime)))
