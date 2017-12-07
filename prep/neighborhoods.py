import os
from rasterio.features import rasterize
from shapely import wkt
from pandas import read_csv
import numpy as np

from model.topoml_util.GeoVectorizer import GeoVectorizer

NEIGHBORHOODS_SOURCE = '../files/neighborhoods/neighborhoods.csv'
NEIGHBORHOODS_TRAIN = '../files/neighborhoods/neighborhoods_train.npz'
NEIGHBORHOODS_TEST = '../files/neighborhoods/neighborhoods_test.npz'
SANE_NUMBER_OF_POINTS = 512
TRAIN_TEST_SPLIT = 0.1

if not os.path.isfile(NEIGHBORHOODS_SOURCE):
    raise FileNotFoundError('Unable to locate %s. Please run the get-data.sh script first' % NEIGHBORHOODS_SOURCE)

with open(NEIGHBORHOODS_SOURCE) as file:
    df = read_csv(file)

print('Creating geometry vectors...')
geoms = [GeoVectorizer.vectorize_wkt(wkt_string, SANE_NUMBER_OF_POINTS, simplify=True) for wkt_string in df.geom.values]

print('Creating geometry fourier descriptors...')
shapes = [wkt.loads(wkt_string).geoms[0] for wkt_string in df.geom.values]
fourier_descriptors = []

for index, shape in enumerate(shapes):
    boundary = shape.boundary
    while boundary.geom_type == "MultiLineString":
        boundary = boundary.geoms[0]
    try:
        fourier_descriptors.append(np.fft.fft(boundary.coords))
    except Exception as e:
        print('Error %s on geom at csv line %i' % (e, index + 2))

print('Saving to numpy archive...')
train_test_split_index = round(TRAIN_TEST_SPLIT * len(geoms))
np.savez(
    NEIGHBORHOODS_TRAIN,
    input_geoms=geoms[:-train_test_split_index],
    inhabitants=df.aantal_inwoners[:-train_test_split_index],
    fourier_descriptors=fourier_descriptors[:-train_test_split_index]
)

np.savez(
    NEIGHBORHOODS_TEST,
    input_geoms=geoms[-train_test_split_index:],
    inhabitants=df.aantal_inwoners[-train_test_split_index:],
    fourier_descriptors=fourier_descriptors[-train_test_split_index:]
)

print('Done!')
