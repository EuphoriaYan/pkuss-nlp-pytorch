from torch import nn
import torch


# Encoder部分，其实就是一个标准的RNN网络
class EncoderRNN(nn.Module):
    # 这里的input_size其实就是英语的词表大小，hidden_size是超参
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 词嵌入层，这里没有初始化，就让它随着训练自己计算吧
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 注意这里用了batch_first，所以接收的输入是(batch_size, seq_length, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        return output, hidden


# Decoder部分，我们这里先用标准的RNN网络，实际上现在大部分是使用带Attention的RNN网络
class DecoderRNN(nn.Module):
    # 这里的output_size其实就是法语的词表大小，hidden_size必须要和刚才Encoder的hidden_size一致
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # 注意这里用了batch_first，所以接收的输入是(batch_size, seq_length, hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden):
        # 输入的x是(batch_size)大小，经过embedding后变成(batch_size, hidden_size)，但还是和GRU的要求不一致
        # 好在decoder中，seq_length始终是1，所以我们只需要用unsqueeze函数，在中间加一维即可
        # (batch_size, hidden_size) -> (batch_size, 1, hidden_size) -> (batch_size, seq_length=1, hidden_size)
        embedded = self.dropout(self.embedding(x).unsqueeze(1))
        # 通过RNN，得到下一个输出，以及输出对应的hidden_state
        output, hidden = self.gru(embedded, hidden)
        # 用fc输出层进行预测
        prediction = self.out(output)
        return prediction, hidden


class Seq2seq_translater(nn.Module):
    # input_size: 英文词表数
    # hidden_size: 超参
    # output_size: 法文词表数
    # seq_length: 最长序列长度
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super().__init__()
        # 用上之前的Encoder
        self.encoder = EncoderRNN(input_size, hidden_size)
        # 用上之前的Decoder
        self.decoder = DecoderRNN(hidden_size, output_size)
        # 小小的注意事项：hidden_size，num_layers必须一致，不然报错
        # 用交叉熵损失函数
        self.loss_fct = nn.CrossEntropyLoss()
        # 这两个属性存下来，还要用到
        self.output_size = output_size
        self.seq_length = seq_length

    def forward(self, x, y=None):
        # 别忘记了我们在GRU里使用了batch_first，所以输入和输出的shape都是batch在最前面的
        # output是(batch_size, seq_length, hidden_size)
        # 但是hidden(以及如果你用LSTM，还多一个cell)，还是原来的样子
        # hidden是(num_layers*num_directions, batch_size, hidden_size)
        output, hidden = self.encoder(x)
        # 用SOS(START OF SENTENCE)来做Decoder的初始输入，我们可以直接从输入中提取
        # 每一轮的输入的shape就是个(batch_size)，一维的
        decoder_input = x[:, 0]
        # 依次存下每轮的输出
        outputs = []
        # 一轮一轮地进行迭代
        for i in range(self.seq_length):
            # decoder迭代一次
            output, hidden = self.decoder(decoder_input, hidden)
            # 存在outputs里面
            outputs.append(output)
            # 这个时候的outputs是(batch_size, 1, output_size)，我们在最后一维上做argmax，就能得到输出的结果
            # 但是别忘记了，输入是(batch_size)，所以我们需要进行一个squeeze，把当中那个1去了
            pred = output.squeeze(dim=1).argmax(dim=-1)
            # 下一轮的输入就是本轮的预测
            decoder_input = pred
        # 最后我们把所有预测的连起来，在当中那维连起来
        total_output = torch.cat(outputs, dim=1)
        # 还是一样，如果有y就输出loss，没有y就输出预测
        if y is not None:
            return self.loss_fct(total_output.view(-1, self.output_size), y.view(-1))
        else:
            return total_output.squeeze()
