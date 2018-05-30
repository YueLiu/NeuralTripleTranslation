
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size,
                 num_layers, vocab_size, num_classes, dropout):
        super(BiLSTM, self).__init__()

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, seq_lengths):
        batch_size = inputs.size(0)

        inputs = self.word_embeds(inputs)

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))

        # Forward propagate RNN
        outputs, _ = self.lstm(inputs, (h0, c0))

        # Decode hidden state of last time step
        outputs = F.relu(self.fc1(outputs[:, -1, :]))

        outputs = self.dropout(outputs)

        outputs = self.fc2(outputs)

        return outputs


class BasicRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size,
                 num_layers, vocab_size, num_classes, dropout):
        super(BasicRNN, self).__init__()

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, seq_lengths):
        batch_size = inputs.size(0)

        inputs = self.word_embeds(inputs)

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))

        # Forward propagate RNN
        outputs, _ = self.rnn(inputs, h0)

        # Decode hidden state of last time step
        outputs = F.relu(self.fc1(outputs[:, -1, :]))

        outputs = self.dropout(outputs)

        outputs = self.fc2(outputs)

        return outputs


class GRURNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size,
                 num_layers, vocab_size, num_classes, dropout):
        super(GRURNN, self).__init__()

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, seq_lengths):
        batch_size = inputs.size(0)

        inputs = self.word_embeds(inputs)

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))

        # Forward propagate RNN
        outputs, _ = self.rnn(inputs, h0)

        # Decode hidden state of last time step
        outputs = F.relu(self.fc1(outputs[:, -1, :]))

        outputs = self.dropout(outputs)

        outputs = self.fc2(outputs)

        return outputs
