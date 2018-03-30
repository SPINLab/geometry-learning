import os
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from model.topoml_util.GeoVectorizer import GeoVectorizer
from model.topoml_util.geom_fourier_descriptors import geom_fourier_descriptors
from pandas import read_csv
from shapely import wkt

from prep.ProgressBar import ProgressBar

SOURCE_ZIP = '../files/neighborhoods/neighborhoods.csv.zip'
SOURCE_CSV = 'neighborhoods.csv'
NEIGHBORHOODS_TRAIN = '../files/neighborhoods/neighborhoods_order_30_train.npz'
NEIGHBORHOODS_TEST = '../files/neighborhoods/neighborhoods_order_30_test.npz'
SANE_NUMBER_OF_POINTS = 512
TRAIN_TEST_SPLIT = 0.1
FOURIER_DESCRIPTOR_ORDER = 30  # The axis 0 size

if not os.path.isfile(SOURCE_ZIP):
    raise FileNotFoundError('Unable to locate %s. Please run the get-data.sh script first' % SOURCE_ZIP)

zfile = ZipFile(SOURCE_ZIP)
df = read_csv(zfile.open(SOURCE_CSV))
df = df[df.aantal_inwoners >= 0]  # Filter out negative placeholder values for unknowns

print('Creating neighborhood geometry vectors...')
pgb = ProgressBar()
geoms = []
for index, wkt_string in enumerate(df.geom.values):
    pgb.update_progress(index/len(df.geom.values))
    geoms.append(GeoVectorizer.vectorize_wkt(wkt_string, SANE_NUMBER_OF_POINTS, simplify=True))

print('Creating neighborhood geometry fourier descriptors...')
shapes = []
for wkt_string in df.geom.values:
    shape = wkt.loads(wkt_string)
    # Out of the 13,300 neighborhoods there's about 300 multipart multipolygon geometries.
    # We're selecting the largest here, but it will throw off the accuracy a bit.
    geometries = sorted(shape.geoms, key=lambda x: x.area)
    shapes.append(geometries[-1])

fourier_descriptors = geom_fourier_descriptors(shapes, FOURIER_DESCRIPTOR_ORDER)

print('Creating categories for neighborhood inhabitants')
# This will create a roughly even (6610:6598) split as the number of inhabitants isn't distributed normally.
median = np.median(df.aantal_inwoners.values)
print('Median:', median, 'inhabitants')
above_or_below_median = []
for number in df.aantal_inwoners.values:
    category = [1, 0] if number > median else [0, 1]
    above_or_below_median.append(category)

# Scatterplot of area versus inhabitants
areas = [shape.area for shape in shapes]
plt.scatter(areas, df.aantal_inwoners.values, s=0.1, alpha=0.5)
plt.xlim(0, 5e-3)
plt.ylim(0, 10000)
# plt.savefig('neighborhood_area_inhabitants_scatter.png')

print('Saving to neighborhoods numpy train and test archives...')
train_test_split_index = round(TRAIN_TEST_SPLIT * len(geoms))
np.savez_compressed(
    NEIGHBORHOODS_TRAIN,
    input_geoms=geoms[:-train_test_split_index],
    inhabitants=df.aantal_inwoners[:-train_test_split_index],
    fourier_descriptors=fourier_descriptors[:-train_test_split_index],
    above_or_below_median=above_or_below_median[:-train_test_split_index],
)

np.savez_compressed(
    NEIGHBORHOODS_TEST,
    input_geoms=geoms[-train_test_split_index:],
    inhabitants=df.aantal_inwoners[-train_test_split_index:],
    fourier_descriptors=fourier_descriptors[-train_test_split_index:],
    above_or_below_median=above_or_below_median[-train_test_split_index:],
)

print('Done!')
