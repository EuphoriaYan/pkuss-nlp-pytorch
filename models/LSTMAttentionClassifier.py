import torch
from torch import nn


class LSTMAttentionClassifierNet(nn.Module):
    def __init__(self, seq_length, label_len, emb_matrix, hidden_dims=None, bidirectional=False, num_layers=1):
        super(LSTMAttentionClassifierNet, self).__init__()
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

        # attention层
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dims, 1),
            nn.ReLU(inplace=True)
        )

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
        # LSTM输出的其实是最后一层的每个时刻的“短时记忆”
        x, (final_h, final_c) = self.lstm(x)
        # 重新把维度换成(batch, seq_len, num_directions * hidden_size)
        x = x.permute(1, 0, 2)

        # 双向的话，我们把两个方向的取和，现在x的形状是(batch, seq_len, hidden_size)
        if self.bidirectional:
            x = torch.chunk(x, 2, -1)
            x = x[0] + x[1]

        # 接下来我们计算attention

        # (batch, seq_len, hidden_size)
        x = nn.Tanh()(x)

        # atten_context (batch_size, seq_len, 1)
        atten_context = self.attention(x)
        atten_context = atten_context.permute(0, 2, 1)
        # softmax_w (batch_size, 1, seq_len)
        softmax_w = self.softmax(atten_context)

        # atten_x (batch_size, 1, hidden_dims)
        atten_x = torch.bmm(softmax_w, x)
        # (batch_size, hidden_dims)
        atten_x = atten_x.squeeze(dim=1)
        logits = self.FC_out(atten_x)
        if y is None:
            return logits
        else:
            return self.loss_fct(logits, y)
