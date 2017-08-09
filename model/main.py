import numpy
import pandas
from keras import Input
from keras.activations import tanh, softmax
from keras.engine import Model
from keras.layers import LSTM

from topoml_util.CustomCallback import CustomCallback
from topoml_util.Tokenizer import Tokenize
from topoml_util.GeoVectorizer import GeoVectorizer, GEO_VECTOR_LEN

# TODO: increase the num_steps in the training set to 10,000,000 (like sketch-rnn)
# TODO: fiddle with the batch size on CUDA cores
# TODO: use recurrent dropout

# To suppress tensorflow info level messages:
# export TF_CPP_MIN_LOG_LEVEL=2

TOPOLOGY_TRAINING_CSV = '../files/topology-training.csv'
MAX_SEQUENCE_LEN = 250
TRAIN_VALIDATE_SPLIT = 0.1
EPOCHS = 60

print('Reading data...')
training_data = pandas.read_csv(TOPOLOGY_TRAINING_CSV)
raw_training_set = training_data['brt_wkt'] + ';' + training_data['osm_wkt']
raw_target_set = training_data['intersection_wkt']
print(len(raw_training_set), 'data points in training set')

(training_set, target_set) = Tokenize.truncate(MAX_SEQUENCE_LEN,
                                               raw_training_set,
                                               raw_target_set)

print(len(target_set), 'max length data points in training set')

brt_wkt = []
osm_wkt = []
for record in training_set:
    sets = record.split(';')
    brt_wkt.append(sets[0])
    osm_wkt.append(sets[1])
max_points = GeoVectorizer.max_points(brt_wkt, osm_wkt)

print('Vectorizing WKT geometries...')
training_vectors = numpy.zeros((len(target_set), max_points, GEO_VECTOR_LEN))
target_vectors = numpy.zeros((len(target_set), max_points, GEO_VECTOR_LEN))

for record_index in range(len(brt_wkt)):
    training_vector = (GeoVectorizer.vectorize_two_wkts(brt_wkt[record_index], osm_wkt[record_index], max_points))
    for point_index, point in enumerate(training_vector):
        for feature_index, feature in enumerate(point):
            training_vectors[record_index][point_index][feature_index] = feature

    target_vector = (GeoVectorizer.vectorize_wkt(training_set[record_index], max_points))
    for point_index, point in enumerate(target_vector):
        for feature_index, feature in enumerate(point):
            target_vectors[record_index][point_index][feature_index] = feature

inputs = Input(shape=(max_points, GEO_VECTOR_LEN))
encoded = LSTM(GEO_VECTOR_LEN, name='Encoding_LSTM', return_sequences=True)(inputs)
encoded = LSTM(GEO_VECTOR_LEN, name='Hidden_LSTM', return_sequences=True)(encoded)
encoded = LSTM(GEO_VECTOR_LEN, name='Hidden_LSTM2', return_sequences=True)(encoded)
# decoded = LSTM(word_index_size, return_sequences=True, name='Decoding_LSTM')(inputs)
encoder = Model(inputs=inputs, outputs=encoded)

encoder.summary()
encoder.compile(loss='mean_squared_error', optimizer='rmsprop')

# encoder = Model(inputs=inputs, outputs=encoded)

# tb_callback = TensorBoard(log_dir='./tensorboard_log', histogram_freq=1, write_graph=True, write_images=True)
my_callback = CustomCallback(GeoVectorizer.decypher)

encoder.fit(x=training_vectors,
            y=target_vectors,
            epochs=EPOCHS,
            batch_size=1000,
            validation_split=TRAIN_VALIDATE_SPLIT,
            callbacks=[my_callback])



