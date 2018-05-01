"""
Preprocessing script to convert well-known-text geometries to matrix representations thereof.
With a SANE_NUMBER_OF_POINTS set to 2048, it simplifies only 248
"""

import os
from datetime import timedelta
from time import time
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from shapely import wkt
from sklearn.model_selection import train_test_split

from model.topoml_util.GeoVectorizer import GeoVectorizer
from model.topoml_util.geom_fourier_descriptors import create_geom_fourier_descriptor
from prep.ProgressBar import ProgressBar

SCRIPT_VERSION = '5'
SOURCE_DIR = '../files/neighborhoods/'
SOURCE_ZIP = SOURCE_DIR + 'neighborhoods.csv.zip'
SOURCE_CSV = 'neighborhoods.csv'
LOG_FILE = 'neighborhoods_preprocessing.log'
TRAIN_DATA_FILE = SOURCE_DIR + 'neighborhoods_train_v' + SCRIPT_VERSION
TEST_DATA_FILE = SOURCE_DIR + 'neighborhoods_test_v' + SCRIPT_VERSION
SANE_NUMBER_OF_POINTS = 2048
TRAIN_TEST_SPLIT = 0.1
FOURIER_DESCRIPTOR_ORDER = 32  # The axis 0 size
SCRIPT_START = time()

if not os.path.isfile(SOURCE_ZIP):
    raise FileNotFoundError('Unable to locate {}. Please run the prep/get-data.sh script first'.format(SOURCE_ZIP))

print('Preprocessing archaeological features...')
zip_file = ZipFile(SOURCE_ZIP)
df = read_csv(zip_file.open(SOURCE_CSV))
df = df[df.aantal_inwoners >= 0]  # Filter out negative placeholder values for unknowns

print('Creating geometry vectors and descriptors...')
wkt_vectors = []
shapes = [wkt.loads(wkt_string) for wkt_string in df.geom.values]
number_of_vertices = [GeoVectorizer.num_points_from_wkt(shape.wkt) for shape in shapes]

plt.hist(number_of_vertices, bins=20, log=True)
plt.savefig('neighborhood_geom_vertices_distr.png')
geoms_above_threshold = len([v for v in number_of_vertices if v > SANE_NUMBER_OF_POINTS])
print('{} of the {} geometries are over the max {} vertices threshold and will be simplified.\n'.format(
    geoms_above_threshold, len(shapes), SANE_NUMBER_OF_POINTS))

pgb = ProgressBar()
logfile = open(LOG_FILE, 'w')
selected_data = []
simplified_geometries = 0
errors = 0
median = np.median(df.aantal_inwoners.values)
print('Median:', median, 'inhabitants')

for index, (inhabitants, wkt_string) in enumerate(zip(df.aantal_inwoners.values, df.geom.values)):
    pgb.update_progress(index/len(df.geom.values), '{} geometries, {} errors in logfile'.format(index, errors))
    try:
        shape = wkt.loads(wkt_string)
        geom_len = min(GeoVectorizer.num_points_from_wkt(shape.wkt), SANE_NUMBER_OF_POINTS)
        if geom_len == SANE_NUMBER_OF_POINTS:
            simplified_geometries += 1
        wkt_vector = GeoVectorizer.vectorize_wkt(wkt_string, geom_len, simplify=True)

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
            logfile.write('skipping record: no (multi)polygon entry in {} on line {}'.format(
                SOURCE_CSV, index + 2))
            errors += 1
            continue

        efds = create_geom_fourier_descriptor(shape, FOURIER_DESCRIPTOR_ORDER)
        # fds = fourierDescriptor(np.array(shape.boundary.coords))

        above_or_below_median = [1, 0] if inhabitants > median else [0, 1]

    except Exception as e:
        logfile.write('Skipping record on account of geometry entry in {} on line {} with error: {}\n'.format(
            SOURCE_CSV, index + 2, e))
        errors += 1
        continue

    # Append the converted values if all went well
    selected_data.append({
        'geom': wkt_vector,
        # 'fourier_descriptors': fds,
        'elliptic_fourier_descriptors': efds,
        'above_or_below_median': above_or_below_median
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
    fourier_descriptors=[record['elliptic_fourier_descriptors'] for record in test],
    above_or_below_median=[record['above_or_below_median'] for record in test])

print('Saving training data...')
np.savez_compressed(
    TRAIN_DATA_FILE,
    geoms=[record['geom'] for record in train],
    fourier_descriptors=[record['elliptic_fourier_descriptors'] for record in train],
    above_or_below_median=[record['above_or_below_median'] for record in train])

runtime = time() - SCRIPT_START
print('Done in {}'.format(timedelta(seconds=runtime)))
