
import collections


class Tokenizer:
    # 初始化的时候读取词表
    def __init__(self, vocab_list):
        self.unk = 'UNK'
        self.vocab = self.load_vocab(vocab_list)
        for i, (k, v) in enumerate(self.vocab.items()):
            if i > 9:
                break
            print(k, v)

    # 读取词表
    def load_vocab(self, vocab_list):
        # 我们一般使用顺序字典来存储词表，这样能够保证历遍时index升序排列
        vocab = collections.OrderedDict()
        # 一般我们使用'UNK'来表示词表中不存在的词，放在0号index上
        vocab[self.unk] = 0
        index = 1
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


if __name__ == '__main__':
    vocab = ["一", "二", "三"]
    tokenizer = Tokenizer(vocab)
