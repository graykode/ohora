import omegaconf
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super().__init__()
        self.conf = conf

        self.num_layers = conf.model.num_layers
        self.hidden_size = conf.model.hidden_size
        self.rnn = nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 3)

    def init_hidden(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        return [t.to(x.device) for t in (h0, c0)]

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (_, _) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
