
import os
os.environ['CUDA_VISIBLE_DEVICE'] = "1"

import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# 个人小建议：系统的包放在上面，个人的包放在下面，这样遇到word_dir出问题的时候中间可以直接修改work_dir

from models.Seq2seqTranslate import Seq2seq_translater
from dataset_readers.trans import *
from utils.tokenizer import Tokenizer


def load_data(seq_length, batch_size):

    data_loader = En2Fr_Trans()
    train_examples = data_loader.get_train_examples()
    en_lines = [Tokenizer.normalizeString(i.text_a) for i in train_examples]
    fr_lines = [Tokenizer.normalizeString(i.text_b) for i in train_examples]

    tokenizer_a = Tokenizer(Tokenizer.get_vocabs(en_lines))
    tokenizer_b = Tokenizer(Tokenizer.get_vocabs(fr_lines))
    word_cnt_a = len(tokenizer_a.vocab)
    word_cnt_b = len(tokenizer_b.vocab)

    def generate_dataloader(examples, tokenizer_a, tokenizer_b, seq_length):
        features = convert_sents_pair(examples, tokenizer_a, tokenizer_b, seq_length)
        text_a = torch.tensor([f.text_a for f in features], dtype=torch.long)
        text_b = torch.tensor([f.text_b for f in features], dtype=torch.long)
        dataset = TensorDataset(text_a, text_b)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    train_dataloader = generate_dataloader(train_examples, tokenizer_a, tokenizer_b, seq_length)
    return train_dataloader, word_cnt_a, word_cnt_b


def load_model(seq_length, word_cnt_a, word_cnt_b):
    model = Seq2seq_translater(word_cnt_a, 256, word_cnt_b, seq_length)
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))
    optimizer = Adam(model.parameters(), lr=0.001)
    return model, optimizer


def train(model, optimizer, train_dataloader, epoch=5):
    total_start_time = time.time()
    for i in range(epoch):
        epoch_start_time = time.time()
        print("epoch %d/%d" % (i + 1, epoch))
        model.train()
        total_loss = []
        for ids, label_ids in train_dataloader:
            if torch.cuda.is_available():
                ids = ids.to(torch.device('cuda'))
                label_ids = label_ids.to(torch.device('cuda'))
            optimizer.zero_grad()
            loss = model(ids, label_ids)
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print("epoch: %d, loss: %.6f" % (i + 1, sum(total_loss) / len(total_loss)))
        epoch_end_time = time.time()
        print("epoch time: %d s" % (epoch_end_time - epoch_start_time))
        torch.save(model.state_dict(), './models/seq2seq_translate.bin')
    total_end_time = time.time()
    print("total time: %d s" % (total_end_time - total_start_time))


if __name__ == '__main__':
    seq_length = 64
    batch_size = 32
    epoch = 10
    train_dataloader, word_cnt_a, word_cnt_b = load_data(seq_length, batch_size)
    model, optimizer = load_model(seq_length, word_cnt_a, word_cnt_b)
    train(model, optimizer, train_dataloader, epoch)
