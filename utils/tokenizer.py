
import collections
import unicodedata
import re


class Tokenizer:
    # 初始化的时候读取词表
    def __init__(self, vocab_list):
        self.unk = 'UNK'
        self.vocab = self.load_vocab(vocab_list)

    # 读取词表
    def load_vocab(self, vocab_list):
        # 我们一般使用顺序字典来存储词表，这样能够保证历遍时index升序排列
        vocab = collections.OrderedDict()
        # 一般我们使用'UNK'来表示词表中不存在的词，放在0号index上
        vocab[self.unk] = 0
        # 在Seq2seq里我们还需要用到“SOS”和“EOS”
        vocab['SOS'] = 1
        vocab['EOS'] = 2
        index = 3
        # 依次插入词
        for token in vocab_list:
            token = token.strip()
            vocab[token] = index
            index += 1
        return vocab

    # 将单个字/词转换为数字id
    def token_to_id(self, token):
        idx = self.vocab.get(token)
        # 不在词表里的词
        if idx is not None:
            return idx
        else:
            return self.vocab[self.unk]

    # 将多个字/词转换为数字id
    def tokens_to_ids(self, tokens):
        ids_list = list(map(self.token_to_id, tokens))
        return ids_list

    @classmethod
    def get_vocabs(cls, lines):
        words = set()
        for line in lines:
            for word in line.strip().split(' '):
                words.add(word)
        return words

    @classmethod
    def unicodeToAscii(cls, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @classmethod
    def normalizeString(cls, s):
        s = cls.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s


if __name__ == '__main__':
    vocab = ["一", "二", "三"]
    tokenizer = Tokenizer(vocab)
