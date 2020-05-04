import torch
from torch import nn


class CNNClassifierNet(nn.Module):
    def __init__(self, seq_length, label_len, emb_matrix):
        super(CNNClassifierNet, self).__init__()
        self.seq_length = seq_length
        self.label_len = label_len
        self.kernel_size = 3
        # 第一层是一个嵌入层，输入为(batch_size, seq_length),输出为(batch_size, seq_length, emb_size)
        # 嵌入层如果使用了from_pretrained，会关掉自动梯度，也就是变得不能训练。如果需要可以手动开启。
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix))
        self.emb_size = self.emb.embedding_dim
        # ReLU层无参数，可以共用
        self.relu = nn.ReLU()

        # 卷积层，输入为(batch_size, emb_size, seq_length)，输出为(batch_size, out_channels, seq_length-self.kernel_size+1)
        self.conv1 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3)
        # 池化层，一般都是一个卷积一个池化，输入为(batch_size, out_channels, seq_length-self.kernel_size+1)
        # 输出为(batch_size, out_channels, 1) -> batch_size, out_channels)
        self.avg1 = nn.AvgPool1d(kernel_size=self.seq_length - self.kernel_size + 1)
        # self.max1 = nn.MaxPool1d(kernel_size=self.seq_length-self.kernel_size+1, return_indices=False)
        # dropout层
        self.dropout = nn.Dropout(p=0.2)

        # 全连接层，输入为(batch_size, out_channels)，输出为(batch_size, 20)
        self.linear2 = nn.Linear(100, 20)
        # 全连接层，输入为(batch_size, 20)，输出为(batch_size, label_len)
        self.linear3 = nn.Linear(20, self.label_len)
        # softmax分类层
        self.softmax = nn.Softmax(dim=-1)
        # 使用交叉熵损失函数
        # 交叉熵损失函数实际上等于nn.Softmax+nn.NLLLoss（负对数似然损失），所以用这个损失的时候不需要先过softmax层
        self.loss = nn.CrossEntropyLoss()

    # forward 定义前向传播，参数不同，输出结果也不同
    def forward(self, x, y=None):
        # 嵌入层，输出为(batch_size, seq_length, emb_size)
        x = self.emb(x)
        # 卷积层需要的输入为(batch_size, emb_size, seq_length)，我们需要将后两维换一下顺序
        # (0, 1, 2)
        x = x.permute(0, 2, 1)
        # 过第一个线性层
        x = self.conv1(x)
        # 过了avg_pooling后大小为(batch_size, channel_size, 1)
        x = self.avg1(x)
        # 我们不需要最后那一维，去掉
        x = x.squeeze_(dim=-1)

        # batch_size, channel_size
        # 非线性激活函数
        x = self.relu(x)
        # 过第二个线性层
        x = self.linear2(x)
        # dropout层
        x = self.dropout(x)
        # 非线性激活函数
        x = self.relu(x)
        # 过第三个线性层
        x = self.linear3(x)

        # 如果没有输入y，那么是在预测，我们返回分类的结果
        if y is None:
            return self.softmax(x)
        # 如果有输入y，那么是在训练，我们返回损失函数的值
        else:
            return self.loss(x, y)