import os
import numpy as np
import matplotlib.pyplot as plt
from model.topoml_util.GeoVectorizer import GeoVectorizer
from pandas import read_csv
from pyefd import elliptic_fourier_descriptors
from shapely import wkt

NEIGHBORHOODS_SOURCE = '../files/neighborhoods/neighborhoods.csv'
NEIGHBORHOODS_TRAIN = '../files/neighborhoods/neighborhoods_train.npz'
NEIGHBORHOODS_TEST = '../files/neighborhoods/neighborhoods_test.npz'
SANE_NUMBER_OF_POINTS = 512
TRAIN_TEST_SPLIT = 0.1
FOURIER_DESCRIPTOR_ORDER = 20  # The axis 0 size

if not os.path.isfile(NEIGHBORHOODS_SOURCE):
    raise FileNotFoundError('Unable to locate %s. Please run the get-data.sh script first' % NEIGHBORHOODS_SOURCE)

with open(NEIGHBORHOODS_SOURCE) as file:
    df = read_csv(file)

df = df[df.aantal_inwoners >= 0]  # Filter out negative placeholder values for unknowns

print('Creating neighborhood geometry vectors...')
geoms = [GeoVectorizer.vectorize_wkt(wkt_string, SANE_NUMBER_OF_POINTS, simplify=True) for wkt_string in df.geom.values]

print('Creating neighborhood geometry fourier descriptors...')
shapes = []
for wkt_string in df.geom.values:
    shape = wkt.loads(wkt_string)
    # Out of the 13,300 neighborhoods there's about 300 multipolygon geometries.
    # We're selecting the largest here, but it will throw off the accuracy a bit.
    geometries = sorted(shape.geoms, key=lambda x: x.area)
    shapes.append(geometries[-1])

fourier_descriptors = []

for index, shape in enumerate(shapes):
    boundary = shape.boundary
    while boundary.geom_type == "MultiLineString":
        boundary = boundary.geoms[0]
    try:
        # Set normalize to false to retain size information.
        non_normalized_coeffs = elliptic_fourier_descriptors(
            boundary.coords, order=FOURIER_DESCRIPTOR_ORDER, normalize=False)
        normalized_coeffs = elliptic_fourier_descriptors(
            boundary.coords, order=FOURIER_DESCRIPTOR_ORDER, normalize=True)
        coeffs = np.append(non_normalized_coeffs, normalized_coeffs)  # without axis this will just create an array
        coeffs = np.append(coeffs, [boundary.area, boundary.length, len(boundary.coords)])
        fourier_descriptors.append(coeffs)
    except Exception as e:
        print('Error %s on geom at csv line %i' % (e, index + 2))
        raise e

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
plt.savefig('neighborhood_area_inhabitants_scatter.png')

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
