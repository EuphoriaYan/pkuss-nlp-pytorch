from utils.data_utils import *


class Zuozhuan_Cws:

    def _read_file(self, file):
        examples = []
        trans = {'[BOS]': 'B', '[IOS]': 'I'}
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('|')
                examples.append(data_example(line[0], [trans[i] for i in line[1].split()]))
        return examples

    def get_train_examples(self):
        train_file_path = 'dataset/cws/zuozhuan/ts.txt'
        return self._read_file(train_file_path)

    def get_dev_examples(self):
        dev_file_path = 'dataset/cws/zuozhuan/tt.txt'
        return self._read_file(dev_file_path)
