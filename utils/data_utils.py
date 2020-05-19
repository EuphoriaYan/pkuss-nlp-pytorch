import typing


# 原始数据和标签
class data_example:
    def __init__(self, text, label):
        self.text = text
        self.label = label


# 处理完毕的数据和标签
class data_feature:
    def __init__(self, ids, label_ids):
        self.ids = ids
        self.label_ids = label_ids


# 一对句子的原始数据
class sents_pair_example:
    def __init__(self, text_a: str, text_b: str):
        self.text_a = text_a
        self.text_b = text_b


# 一对句子的处理后的数据
class sents_pair_feature:
    def __init__(self, text_a: list, text_b: list):
        self.text_a = text_a
        self.text_b = text_b


# 处理原始数据
def convert_example_to_feature(examples, tokenizer, seq_length):
    features = []
    for i in examples:
        # 使用tokenizer将字符串转换为数字id
        ids = tokenizer.tokens_to_ids(i.text)
        # 我们规定了最大长度，超过了就切断，不足就补齐（一般补unk，也就是这里的[0]，也有特殊补位符[PAD]之类的）
        if len(ids) > seq_length:
            ids = ids[0: seq_length]
        else:
            ids = ids + [0] * (seq_length - len(ids))
        # 如果这个字符串全都不能识别，那就放弃掉
        if sum(ids) == 0:
            continue
        assert len(ids) == seq_length
        # 处理标签，正面为1，负面为0
        if i.label == 'positive':
            label_ids = 1
        else:
            label_ids = 0
        features.append(data_feature(ids, label_ids))
    return features


# 处理原始数据
def convert_sents_pair(sents_pair, tokenizer_a, tokenizer_b, seq_length):
    features = []
    for i in sents_pair:
        # 使用tokenizer将字符串转换为数字id

        ids_a = tokenizer_a.tokens_to_ids(['BOS'] + tokenizer_a.normalizeString(i.text_a).split() + ['EOS'])
        ids_b = tokenizer_b.tokens_to_ids(tokenizer_b.normalizeString(i.text_b).split() + ['EOS'])

        # 我们规定了最大长度，超过了就切断，不足就补齐（一般补unk，也就是这里的[0]，也有特殊补位符[PAD]之类的）
        if len(ids_a) > seq_length:
            ids_a = ids_a[0: seq_length]
        else:
            ids_a = ids_a + [0] * (seq_length - len(ids_a))
        if len(ids_b) > seq_length:
            ids_b = ids_b[0: seq_length]
        else:
            ids_b = ids_b + [0] * (seq_length - len(ids_b))
        # 如果这个字符串全都不能识别，那就放弃掉
        if sum(ids_a) == 0 or sum(ids_b) == 0:
            continue
        assert len(ids_a) == seq_length
        assert len(ids_b) == seq_length
        features.append(sents_pair_feature(ids_a, ids_b))
    return features
