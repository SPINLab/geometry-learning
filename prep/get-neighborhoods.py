import requests
from pandas import read_csv
from io import StringIO
import numpy as np

from model.topoml_util.GeoVectorizer import GeoVectorizer

WFS_URL = "https://geodata.nationaalgeoregister.nl/wijkenbuurten2017/wfs"

QUERY_STRING = {
    "request": "GetFeature",
    "service": "WFS",
    "version": "2.0.0",
    "typeName": "cbs_buurten_2017",
    "outputFormat": "csv",
    "srsName": "EPSG:4326",
    "PropertyName": "aantal_inwoners,geom"
}

NEIGHBORHOODS_TRAIN = '../files/neighborhoods_train.npz'
NEIGHBORHOODS_TEST = '../files/neighborhoods_test.npz'
SANE_NUMBER_OF_POINTS = 512
TRAIN_TEST_SPLIT = 0.1

print('Getting data...')
response = requests.request("GET", WFS_URL, params=QUERY_STRING)
buffer = StringIO(response.text)
df = read_csv(buffer)

print('Vectorizing to numpy archive...')
geoms = [GeoVectorizer.vectorize_wkt(wkt, SANE_NUMBER_OF_POINTS, simplify=True) for wkt in df.geom.values]

train_test_split_index = round(TRAIN_TEST_SPLIT * len(geoms))
np.savez(
    NEIGHBORHOODS_TRAIN,
    input_geoms=geoms[:-train_test_split_index],
    inhabitants=df.aantal_inwoners[:-train_test_split_index]
)

np.savez(
    NEIGHBORHOODS_TEST,
    input_geoms=geoms[-train_test_split_index:],
    inhabitants=df.aantal_inwoners[-train_test_split_index:]
)

print('Done!')
