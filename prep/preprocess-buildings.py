# 22007 buildings for gatherings
# 23000 industrial buildings
# 23000 building for lodging
# 23000 buildings for habitation
# 23000 shopping buildings
# 21014 office buildings
# 7832 buildings for health care
# 10717 educational buildings
# 6916 buildings for sports facilities

import os
import re
from datetime import timedelta
from time import time
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv, concat
from shapely import wkt
from sklearn.model_selection import train_test_split

from model.topoml_util.GeoVectorizer import GeoVectorizer
from model.topoml_util.geom_fourier_descriptors import create_geom_fourier_descriptor
from prep.ProgressBar import ProgressBar

SCRIPT_VERSION = '5'
SANE_NUMBER_OF_POINTS = 2048
TRAIN_TEST_SPLIT = 0.1
FOURIER_DESCRIPTOR_ORDER = 32  # The axis 0 size
DATA_TYPE = 'buildings'
LOG_FILE = '{}_preprocessing.log'.format(DATA_TYPE)
SOURCE_ZIP = '../files/{}/{}.csv.zip'.format(DATA_TYPE, DATA_TYPE)
TRAIN_DATA_FILE = '../files/{}/{}_train_v{}'.format(DATA_TYPE, DATA_TYPE, SCRIPT_VERSION)
TEST_DATA_FILE = '../files/{}/{}_test_v{}'.format(DATA_TYPE, DATA_TYPE, SCRIPT_VERSION)
SCRIPT_START = time()

building_types = [
    'bijeenkomstfunctie',  # gatherings
    'industriefunctie',  # industrial
    'logiesfunctie',  # lodging
    'woonfunctie',  # habitation
    'winkelfunctie',  # shopping
    'kantoorfunctie',  # office
    'gezondheidszorgfunctie',  # health care
    'onderwijsfunctie',  # educational
    'sportfunctie',  # sports
]

if not os.path.isfile(SOURCE_ZIP):
    raise FileNotFoundError('Unable to locate {}. Please run the get-data.sh script first'.format(SOURCE_ZIP))

zip_file = ZipFile(SOURCE_ZIP)
df = []

print('Reading source data files...')
for f_index, function_type in enumerate(building_types):
    file = 'buildings-' + function_type + '.csv'
    if not len(df):
        df = read_csv(zip_file.open(file))
    else:
        df = concat([df, (read_csv(zip_file.open(file)))])

shapes = [wkt.loads(wkt_string) for wkt_string in df.geometrie.values]
number_of_vertices = [len(re.findall('\d \d', shape.wkt)) for shape in shapes]

vertices_distr_png = 'buildings_geom_vertices_distr.png'
print('Saving histogram of vertices per geometry {}'.format(vertices_distr_png))
plt.hist(number_of_vertices, bins=20, log=True)
plt.savefig(vertices_distr_png)

logfile = open(LOG_FILE, 'w')
selected_data = []
simplified_geometries = 0
errors = 0

print('Processing data...')
pgb = ProgressBar()

for index, (wkt_string, building_type) in enumerate(zip(df.geometrie.values, df.gebruiksdoel.values)):
    pgb.update_progress(index/len(df.geometrie.values))
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
            logfile.write('Skipping record: no (multi)polygon on line {}'.format(index + 2))
            errors += 1
            continue

        efds = create_geom_fourier_descriptor(shape, FOURIER_DESCRIPTOR_ORDER)

        # Label as numerical index
        type_int = building_types.index(building_type)

    except Exception as e:
        logfile.write('Skipping record on account of faulty geometry entry {} with error: {}\n'.format(index + 2, e))
        errors += 1
        continue

    # Append the converted values if all went well
    selected_data.append({
        'geom': wkt_vector,
        'elliptic_fourier_descriptors': efds,
        'building_type': type_int
    })

logfile.close()
print('\ncreated {} data points with {} simplified geometries and {} errors'.format(
    len(selected_data), simplified_geometries, errors))

# Split and save data
train, test = train_test_split(selected_data, test_size=0.1, random_state=42)

print('Saving training data...')
np.savez_compressed(
    TRAIN_DATA_FILE,
    geoms=[record['geom'] for record in train],
    elliptic_fourier_descriptors=[record['elliptic_fourier_descriptors'] for record in train],
    building_type=[record['building_type'] for record in train])

print('Saving test data...')
np.savez_compressed(
    TEST_DATA_FILE,
    geoms=[record['geom'] for record in test],
    elliptic_fourier_descriptors=[record['elliptic_fourier_descriptors'] for record in test],
    building_type=[record['building_type'] for record in test])

runtime = time() - SCRIPT_START
print('Done in {}'.format(timedelta(seconds=runtime)))
