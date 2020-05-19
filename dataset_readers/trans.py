from utils.data_utils import *


class En2Fr_Trans:
    def __init__(self, small=False):
        self.small = small

    def _read_file(self, file):
        examples = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                examples.append(sents_pair_example(line[0], line[1]))
        return examples

    def get_train_examples(self):
        if self.small:
            train_file_path = 'dataset/translate/eng-fra-small.txt'
        else:
            train_file_path = 'dataset/translate/eng-fra.txt'
        return self._read_file(train_file_path)
