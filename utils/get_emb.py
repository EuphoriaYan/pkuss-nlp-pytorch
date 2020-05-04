import collections
import numpy as np
from utils.tokenizer import Tokenizer


def get_emb():
    # 读取切分好的一行，返回词和词向量（numpy的矩阵）
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    with open('素材\sgns.wiki.char', 'r', encoding='utf-8') as emb_file:
        # 文件的开头是词表长度和词嵌入维度
        dict_length, emb_size = emb_file.readline().rstrip().split()
        print('dict_length: ', dict_length)
        print('emb_size: ', emb_size)
        dict_length, emb_size = int(dict_length), int(emb_size)
        # 对每一行做处理，结果存到顺序词典中
        emb = collections.OrderedDict(get_coefs(*line.rstrip().split()) for line in emb_file.readlines())
    return emb, dict_length, emb_size


def get_emb_matrix(emb: collections.OrderedDict,
                   tokenizer: Tokenizer,
                   dict_length: int,
                   emb_size: int) -> np.ndarray:
    # 生成一个全0矩阵，大小为（词典长度+1，嵌入维度）
    emb_matrix = np.zeros((1 + dict_length, emb_size), dtype='float32')

    for word, id in tokenizer.vocab.items():
        emb_vector = emb.get(word)
        if emb_vector is not None:
            # 将编号为id的词的词向量放在id行上
            emb_matrix[id] = emb_vector
    return emb_matrix
