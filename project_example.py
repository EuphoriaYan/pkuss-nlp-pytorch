

import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score

# 个人小建议：系统的包放在上面，个人的包放在下面，这样遇到word_dir出问题的时候中间可以直接修改work_dir
from utils.tokenizer import Tokenizer
from utils.get_emb import *
from models.LSTMClassifier import LSTMClassifierNet
from models.CNNClassifier import CNNClassifierNet
from models.LSTMAttentionClassifier import LSTMAttentionClassifierNet
from dataset_readers.single_sent_clf import *


def load_data(seq_length):
    emb, dict_length, emb_size = get_emb()
    tokenizer = Tokenizer(emb.keys())
    emb_matrix = get_emb_matrix(emb, tokenizer, dict_length, emb_size)

    data_loader = ChnSentiCorp_Clf()
    train_examples = data_loader.get_train_examples()
    dev_examples = data_loader.get_dev_examples()

    def generate_dataloader(examples, tokenizer, seq_length):
        features = convert_example_to_feature(examples, tokenizer, seq_length)
        ids = torch.tensor([f.ids for f in features], dtype=torch.long)
        label = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(ids, label)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        return dataloader

    train_dataloader = generate_dataloader(train_examples, tokenizer, seq_length)
    dev_dataloader = generate_dataloader(dev_examples, tokenizer, seq_length)

    return emb_matrix, train_dataloader, dev_dataloader


def load_model(seq_length, label_len, emb_matrix):
    # TODO: you can choose different model
    # model = CNNClassifierNet(seq_length, label_len, emb_matrix)
    # model = LSTMClassifierNet(seq_length, label_len, emb_matrix, bidirectional=True)
    model = LSTMAttentionClassifierNet(seq_length, label_len, emb_matrix, bidirectional=True)
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))
    optimizer = Adam(model.parameters(), lr=0.001)
    return model, optimizer


def train(model, optimizer, train_dataloader, dev_dataloader, epoch=5):
    for i in range(epoch):
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

        model.eval()
        total_gold = []
        total_pred = []
        for ids, label_ids in dev_dataloader:
            if torch.cuda.is_available():
                ids = ids.to(torch.device('cuda'))
            logits = model(ids)
            logits = logits.detach().cpu().numpy()
            logits = np.argmax(logits, axis=-1)
            logits = logits.tolist()
            total_pred.extend(logits)
            label_ids = label_ids.numpy().tolist()
            total_gold.extend(label_ids)
        # eval_p = precision_score(total_gold, total_pred)
        # eval_r = recall_score(total_gold, total_pred)
        eval_f1 = f1_score(total_gold, total_pred)
        print("eval_f1: %.2f%%" % (eval_f1 * 100))


if __name__ == '__main__':
    seq_length = 30
    label_len = 2
    emb_matrix, train_dataloader, dev_dataloader = load_data(seq_length)
    model, optimizer = load_model(seq_length, label_len, emb_matrix)
    train(model, optimizer, train_dataloader, dev_dataloader, epoch=5)
