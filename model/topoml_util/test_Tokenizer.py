import unittest
import pandas
from Tokenizer import Tokenize

TOPOLOGY_TRAINING_CSV = 'test_files/example.csv'
source_data = pandas.read_csv(TOPOLOGY_TRAINING_CSV)
raw_training_set = source_data['brt_wkt'] + ' ' + source_data['osm_wkt']
raw_target_set = source_data['intersection_wkt']


class TestUtil(unittest.TestCase):
    def test_truncate(self):
        max_len = 500
        (input_set, _) = Tokenize.truncate(max_len, raw_training_set, raw_target_set)
        for record in input_set:
            for field in record:
                self.assertLessEqual(len(field), max_len)

    def test_batch_truncate(self):
        batch_size = 3
        max_len = 1000
        validation_split = 0.1
        training_set, target_set = Tokenize.batch_truncate(batch_size, max_len, validation_split, raw_training_set,
                                                           raw_target_set)
        self.assertEqual(len(training_set), 30)

    def test_tokenize(self):
        test_strings = ['A test string']
        tokenizer = Tokenize(test_strings)
        tokenized = tokenizer.char_level_tokenize(test_strings)
        self.assertEqual((tokenizer.word_index, tokenized),
                         ({' ': 2, 'A': 4, 'e': 5, 'g': 9, 'i': 7, 'n': 8, 'r': 6, 's': 3, 't': 1},
                          [[4, 2, 1, 5, 3, 1, 2, 3, 1, 6, 7, 8, 9]]))

    def test_tokenize_example(self):
        self.maxDiff = None
        test_strings = source_data.as_matrix()
        word_index = {'5': 1, '4': 2, '.': 3, '1': 4, '2': 5, '8': 6, ' ': 7, ',': 8, '3': 9, '6': 10, '0': 11,
                      '9': 12, '7': 13, 'O': 14, '(': 15, ')': 16, 'L': 17, 'Y': 18, 'P': 19, 'G': 20, 'N': 21,
                      'T': 22, 'E': 23, 'M': 24, 'I': 25, 'C': 26, 'U': 27, 'R': 28}
        tokenizer = Tokenize(test_strings[0] + test_strings[1] + test_strings[2])
        tokenized = tokenizer.char_level_tokenize(test_strings[0])
        self.assertEqual((tokenizer.word_index, tokenized[0][0:15]),
                         (word_index,
                          [19, 14, 17, 18, 20, 14, 21, 15, 15, 2, 3, 6, 4, 4, 6]))

    def test_one_hot(self):
        source_matrix = source_data.as_matrix()
        test_strings = source_matrix[0] + source_matrix[1]

        max_len = 0
        for sentence in test_strings:
            if len(sentence) > max_len:
                max_len = len(sentence)

        tokenizer = Tokenize(test_strings)
        matrix = tokenizer.one_hot(test_strings, max_len)
        self.assertEqual(matrix[0][0][19], True)  # 'P' for POLYGON

    def test_detokenize(self):
        test_strings = ['A test string']
        tokenizer = Tokenize(test_strings)
        tokenized = tokenizer.char_level_tokenize(test_strings)
        detokenized = tokenizer.decypher(tokenized)
        self.assertEqual(detokenized, test_strings)
