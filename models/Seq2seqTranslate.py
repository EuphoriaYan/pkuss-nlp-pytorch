from torch import nn
import torch
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden):
        embedded = self.dropout(self.embedding(x).unsqueeze(1))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.out(output)
        return prediction, hidden


class Seq2seq_translater(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(Seq2seq_translater, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, output_size)
        self.loss_fct = nn.CrossEntropyLoss()
        self.output_size = output_size
        self.seq_length = seq_length

    def forward(self, x, y=None):
        output, hidden = self.encoder(x) # output是(batch_size, seq_length, hidden_size), hidden是(num_layers*num_direction, batch_size, hidden_size)
        # 用SOS来做Decoder的初始输入
        decoder_input = x[:, 0]
        outputs = []
        for i in range(self.seq_length):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs.append(output)
            pred = output.squeeze().argmax(dim=-1)
            decoder_input = pred

        total_output = torch.cat(outputs, dim=1)
        if y is not None:
            return self.loss_fct(total_output.view(-1, self.output_size), y.view(-1))
        else:
            return total_output.detach().cpu().numpy().argmax(dim=-1)