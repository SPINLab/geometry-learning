import os

from model.topoml_util.geom_fourier_descriptors import geom_fourier_descriptors
from model.topoml_util.GeoVectorizer import GeoVectorizer
from pandas import read_csv
from shapely import wkt
import numpy as np

SOURCE = '../files/archaeology/combined_arch_feat.csv'
SANE_NUMBER_OF_POINTS = 256
TRAIN_TEST_SPLIT = 0.1
FOURIER_DESCRIPTOR_ORDER = 16  # The axis 0 size
TRAIN_DATA_FILE = '../files/archaeology/archaeology-train-'
TEST_DATA_FILE = '../files/archaeology/archaeology-test.npz'
NUMBER_OF_FILES = 4

if not os.path.isfile(SOURCE):
    raise FileNotFoundError('Unable to locate %s. Please run the prep/get-data.sh script first' % SOURCE)

with open(SOURCE) as file:
    df = read_csv(file)

df = df[df.Aardspoor != '']  # Filter out negative placeholder values for unknowns
