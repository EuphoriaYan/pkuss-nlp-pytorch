import torch
from torch import nn


class LSTMClassifierNet(nn.Module):
    def __init__(self, seq_length, label_len, emb_matrix, hidden_dims=None, bidirectional=False, num_layers=1):
        super(LSTMClassifierNet, self).__init__()
        self.seq_length = seq_length
        self.label_len = label_len
        # 控制是否使用双向LSTM
        self.bidirectional = bidirectional
        if num_layers == 1:
            self.lstm_dropout = 0.0
        else:
            self.lstm_dropout = 0.2
        self.fc_dropout = 0.1
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix))
        self.emb_size = self.emb.embedding_dim
        if hidden_dims is not None:
            self.hidden_dims = hidden_dims
        else:
            self.hidden_dims = self.emb_size

        # 循环神经网络，输入为(seq_len, batch, input_size)，(h_0, c_0), 如果没有给出h_0和c_0则默认为全零
        # 输出为(seq_len, batch, num_directions * hidden_size), (h_final, c_final)
        # 关于hidden_state和cell_state，可以理解为“短期记忆”和“长期记忆”
        self.lstm = nn.LSTM(self.emb_size, self.hidden_dims,
                            num_layers=1, dropout=self.lstm_dropout,
                            bidirectional=self.bidirectional)

        # 输出层，输入为(batch_size, hidden_dims)，输出为(batch_size, label_len)
        self.FC_out = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.hidden_dims, self.label_len)
        )

        # softmax分类层
        self.softmax = nn.Softmax(dim=-1)
        # 交叉熵损失函数
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 通过词嵌入得到词的分布式表示，输出是(batch_size, seq_len, input_size)
        x = self.emb(x)
        # 但是LSTM要的输入是(seq_len, batch_size, input_size)，做一下维度变换
        # 你也可以在建立LSTM网络的时候设置"batch_first = True"，使得LSTM要的输入就是(batch_size, seq_len, input_size)
        x = x.permute(1, 0, 2)
        # 使用LSTM，输出为(seq_len, batch_size, num_directions * hidden_size)
        # LSTM输出的其实是最后一层的每个时刻的“短期记忆”
        x, (final_h, final_c) = self.lstm(x)
        # 我们就用最终的“长期记忆”来做分类，也就是final_c，它的维度是: (num_layers * num_directions, batch_size, hidden_size)
        # 我们把batch_size放到最前面，所以现在是(batch_size, num_layers * num_directions, hidden_size)
        final_c = final_c.permute(1, 0, 2)

        # 把每一层和每个方向的取个平均值，变成(batch_size, hidden_size)，现在就可以去做FC操作了
        final_c = final_c.mean(dim=1)

        logits = self.FC_out(final_c)
        if y is None:
            return logits
        else:
            return self.loss_fct(logits, y)