from utils.data_utils import *
from sklearn.model_selection import train_test_split


class ChnSentiCorp_Clf:

    def __init__(self):
        # 读原始数据
        self.examples = []
        with open('素材/sentiment/正面评价.txt', 'r', encoding='utf-8') as pos_file:
            for line in pos_file:
                line = line.strip()
                self.examples.append(data_example(line, 'positive'))
        with open('素材/sentiment/负面评价.txt', 'r', encoding='utf-8') as neg_file:
            for line in neg_file:
                line = line.strip()
                self.examples.append(data_example(line, 'negative'))
        self.train_set, self.dev_set = train_test_split(self.examples, test_size=0.1, random_state=777)

    def get_train_examples(self):
        return self.train_set

    def get_dev_examples(self):
        return self.dev_set

