"""
Preprocessing script to convert well-known-text geometries to matrix representations thereof.
With a SANE_NUMBER_OF_POINTS set to 2048, it simplifies only 248
"""

import os
import re
from datetime import timedelta
from time import time
from zipfile import ZipFile

import collections
import numpy as np
from pandas import read_csv
from shapely import wkt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model.topoml_util.GeoVectorizer import GeoVectorizer
from model.topoml_util.geom_fourier_descriptors import create_geom_fourier_descriptor
from prep.ProgressBar import ProgressBar

SCRIPT_VERSION = '5'
DATA_FOLDER = '../files/archaeology/'
SOURCE_ZIP = DATA_FOLDER + 'archaeology.csv.zip'
SOURCE_CSV = 'archaeology.csv'
LOG_FILE = 'archaeology_preprocessing.log'
TRAIN_DATA_FILE = DATA_FOLDER + 'archaeology_train_v' + SCRIPT_VERSION
TEST_DATA_FILE = DATA_FOLDER + 'archaeology_test_v' + SCRIPT_VERSION
SANE_NUMBER_OF_POINTS = 2048
TRAIN_TEST_SPLIT = 0.1
FOURIER_DESCRIPTOR_ORDER = 32  # The axis 0 size
MINIMUM_CLASS_OCCURRENCE = 1000
SCRIPT_START = time()

if not os.path.isfile(SOURCE_ZIP):
    raise FileNotFoundError('Unable to locate {}. Please run the prep/get-data.sh script first'.format(SOURCE_ZIP))

print('Preprocessing archaeological features...')
zfile = ZipFile(SOURCE_ZIP)
df = read_csv(zfile.open(SOURCE_CSV))

aardspoor__as_matrix = df['Aardspoor'].as_matrix()
wkt__as_matrix = df['WKT'].as_matrix()

class_count = dict(collections.Counter([f_type for f_type in aardspoor__as_matrix if type(f_type) == str]))
included_classes = [f_type for f_type, count in class_count.items() if count > MINIMUM_CLASS_OCCURRENCE]
print('Included classes:', included_classes)

# geometry vectors
print('Creating geometry vectors and descriptors...')
feature_types = []
wkt_vectors = []
fourier_descriptors = []
shapes = []

for wkt_string in df.WKT.values:
    try:
        shapes.append(wkt.loads(wkt_string))
    except Exception as e:
        print('Skipping unreadable wkt geom.')
number_of_vertices = [len(re.findall('\d \d', shape.wkt)) for shape in shapes]

plt.hist(number_of_vertices, bins=20, log=True)
plt.savefig('archaeology_geom_vertices_distr.png')
geoms_above_threshold = len([v for v in number_of_vertices if v > SANE_NUMBER_OF_POINTS])
print('{} of the {} geometries are over the max {} vertices threshold and will be simplified.\n'.format(
    geoms_above_threshold, len(shapes), SANE_NUMBER_OF_POINTS))

pgb = ProgressBar()
logfile = open(LOG_FILE, 'w')
selected_data = []
simplified_geometries = 0
errors = 0

for index, (feature, geom) in enumerate(zip(aardspoor__as_matrix, wkt__as_matrix)):
    pgb.update_progress(index/len(aardspoor__as_matrix), '{} geometries, {} errors in logfile'.format(index, errors))
    if feature in included_classes:
        try:
            shape = wkt.loads(geom)
            geom_len = min(len(re.findall('\d \d', shape.wkt)), SANE_NUMBER_OF_POINTS)
            if geom_len == SANE_NUMBER_OF_POINTS:
                simplified_geometries += 1
            wkt_vector = GeoVectorizer.vectorize_wkt(geom, geom_len, simplify=True)

            # If multipart multipolygon: select the largest, but it will throw off the accuracy a bit.
            if shape.geom_type == 'MultiPolygon':
                if len(shape.geoms) > 1:
                    geometries = sorted(shape.geoms, key=lambda x: x.area)
                    shape = geometries[-1]
                else:
                    shape = shape.geoms[0]
            elif shape.geom_type == 'Polygon':
                pass
            else:
                print('skipping record: no (multi)polygon entry in {} on line {}'.format(
                    SOURCE_CSV, index + 2))
                errors += 1
                continue

            efds = create_geom_fourier_descriptor(shape, FOURIER_DESCRIPTOR_ORDER)

        except Exception as e:
            logfile.write('Skipping record on account of geometry entry in {0} on line {1} with error: {2}\n'.format(
                SOURCE_CSV, index + 2, e))
            errors += 1
            continue

        # Append the converted values if all went well
        selected_data.append({
                'geom': wkt_vector,
                'elliptic_fourier_descriptors': efds,
                'feature_type': included_classes.index(feature),  # Convert types to numerical index
            })

logfile.close()
print('\ncreated {} data points with {} simplified geometries and {} errors'.format(
    len(selected_data), simplified_geometries, errors))

# Split and save data
train, test = train_test_split(selected_data, test_size=0.1, random_state=42)

print('Saving test data...')
# Test data is small enough to put in one archive
np.savez_compressed(
    TEST_DATA_FILE,
    geoms=[record['geom'] for record in test],
    elliptic_fourier_descriptors=[record['elliptic_fourier_descriptors'] for record in test],
    feature_type=[record['feature_type'] for record in test],
    feature_type_index=included_classes)

print('Saving training data...')
np.savez_compressed(
    TRAIN_DATA_FILE,
    geoms=[record['geom'] for record in train],
    elliptic_fourier_descriptors=[record['elliptic_fourier_descriptors'] for record in train],
    feature_type=[record['feature_type'] for record in train],
    feature_type_index=included_classes)

runtime = time() - SCRIPT_START
print('Done in {}'.format(timedelta(seconds=runtime)))
