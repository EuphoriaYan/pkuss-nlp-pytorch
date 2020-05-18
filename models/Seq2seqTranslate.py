from torch import nn
import torch
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output)
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=128):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(embedded), dim=2)
        attn_applied = torch.bmm(attn_weights,
                                 encoder_outputs)

        output = torch.cat((embedded, attn_applied), dim=2)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output)

        output = self.out(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden, attn_weights


class Seq2seq_translater(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_seq_length):
        super(Seq2seq_translater, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, output_size, max_length=max_seq_length)
        self.loss_fct = nn.NLLLoss()
        self.output_size = output_size

    def forward(self, input, y=None):
        x, _ = self.encoder(input)
        x, _, _ = self.decoder(input, x)

        if y is not None:
            return self.loss_fct(x.view(-1, self.output_size), y.view(-1))
        else:
            return x