import numpy as np

from topoml_util.GeoVectorizer import GeoVectorizer
from topoml_util.geom_scaler import localized_mean, localized_normal

DATA_FILE = '../files/brt_osm/brt_osm.npz'
TARGET_FILE = '../files/brt_osm/densified_vectorized.npz'
DENSIFIED = 100

loaded = np.load(DATA_FILE)
raw_training_vectors = loaded['input_geoms']
raw_target_vectors = loaded['intersection']

training_vectors = []
target_vectors = []

# skip non-intersecting geometries
for train, target in zip(raw_training_vectors, raw_target_vectors):
    if not target[0, 0] == 0:  # a zero coordinate designates an empty geometry
        training_vectors.append(train)
        target_vectors.append(target)

print('Preprocessing vectors...')
means = localized_mean(training_vectors)
training_vectors = localized_normal(training_vectors, means, 1e4)
training_vectors = np.array([GeoVectorizer.interpolate(vector, DENSIFIED) for vector in training_vectors])
target_vectors = localized_normal(target_vectors, means, 1e4)
target_vectors = np.array([GeoVectorizer.interpolate(vector, 50) for vector in target_vectors])

print('Saving compressed numpy data file', TARGET_FILE)

np.savez_compressed(
    TARGET_FILE,
    input_geoms=training_vectors,
    intersection=target_vectors
)

print('Done!')
