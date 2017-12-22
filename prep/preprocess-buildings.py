# 22007 buildings for gatherings
# 7832 buildings for health care
# 23000 industrial buildings
# 21014 office buildings
# 23000 building for staying overnight
# 10717 educational buildings
# 6916 buildings for physical exercise
# 23000 shopping buildings
# 23000 buildings for habitation
import os

from model.topoml_util.geom_fourier_descriptors import geom_fourier_descriptors
from model.topoml_util.GeoVectorizer import GeoVectorizer
from pandas import read_csv
from shapely import wkt
import numpy as np

SANE_NUMBER_OF_POINTS = 64
TRAIN_TEST_SPLIT = 0.1
FOURIER_DESCRIPTOR_ORDER = 16  # The axis 0 size
TRAIN_DATA_FILE = '../files/buildings/buildings-train-'
TEST_DATA_FILE = '../files/buildings/buildings-test.npz'
NUMBER_OF_FILES = 6

building_types = [
    'bijeenkomstfunctie',
    'gezondheidszorgfunctie',
    'industriefunctie',
    'kantoorfunctie',
    'logiesfunctie',
    'onderwijsfunctie',
    'sportfunctie',
    'winkelfunctie',
    'woonfunctie',
]

training_data = {
    'geoms': [],
    'fourier_descriptors': [],
    'building_type': [],
}

test_data = {
    'geoms': [],
    'fourier_descriptors': [],
    'building_type': [],
}

for builing_type in building_types:
    path = '../files/buildings/buildings-' + builing_type + '.csv'
    print('Processing', path)

    if not os.path.isfile(path):
        raise FileNotFoundError('Unable to locate %s. Please run the get-data.sh script first' % path)

    with open(path) as data_file:
        df = read_csv(data_file)

    print('Creating building geometry vectors...')
    geoms = []
    for index, wkt_string in enumerate(df.geometrie.values):
        try:
            geoms.append(GeoVectorizer.vectorize_wkt(wkt_string, SANE_NUMBER_OF_POINTS, simplify=True))
        except Exception as e:
            raise ValueError('Incorrect geometry entry in {0} on line {1}: {2} with error {3}'.format(
                path,
                index + 2,
                wkt_string,
                e))

    print('Creating building geometry fourier descriptors...')
    shapes = []
    for wkt_string in df.geometrie.values:
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
    train_test_split_index = round(TRAIN_TEST_SPLIT * len(geoms))

    # Labels
    type_int = [building_types.index(building_type) for building_type in df.gebruiksdoel]

    training_data['geoms'].append(geoms[:-train_test_split_index])
    training_data['fourier_descriptors'].append(fourier_descriptors[:-train_test_split_index])
    training_data['building_type'].append(type_int[:-train_test_split_index])

    test_data['geoms'].append(geoms[-train_test_split_index:])
    test_data['fourier_descriptors'].append(fourier_descriptors[-train_test_split_index:])
    test_data['building_type'].append(type_int[-train_test_split_index:])

print('Saving training and test data files...')

part_size = round(len(training_data['geoms']) / NUMBER_OF_FILES)

# Create git-sized chunks
for part in range(NUMBER_OF_FILES):
    start_index = part_size * part
    stop_index = part_size * (part + 1)
    section = {
        'geoms': training_data['geoms'][start_index:stop_index],
        'fourier_descriptors': training_data['fourier_descriptors'][start_index:stop_index],
        'building_type': training_data['building_type'][start_index:stop_index]
    }

    np.savez_compressed(
        TRAIN_DATA_FILE + str(part),
        GEOMS=section['geoms'],
        FOURIER_DESCRIPTORS=section['fourier_descriptors'],
        BUILDING_TYPE=section['building_type'],
    )

# Test data is small enough to put in one archive
np.savez_compressed(
    TEST_DATA_FILE,
    GEOMS=test_data['geoms'],
    FOURIER_DESCRIPTORS=test_data['fourier_descriptors'],
    BUILDING_TYPE=test_data['building_type'],
)

print('Done!')
