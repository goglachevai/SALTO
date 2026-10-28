import torch
from torch import nn

KERNEL = 3


class Encoder(nn.Module):
    def __init__(self, conv1=128, conv2=64, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv1, KERNEL), nn.ReLU(),
            nn.Conv1d(conv1, conv2, KERNEL), nn.ReLU(),
        )
        self.gru = nn.GRU(conv2, hidden, batch_first=True)

    def forward(self, x):
        h = self.conv(x.unsqueeze(1))
        _, h_last = self.gru(h.transpose(1, 2))
        return h_last


class Decoder(nn.Module):
    def __init__(self, m, conv1=128, conv2=64, hidden=32):
        super().__init__()
        self.steps = m - 2 * (KERNEL - 1)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(hidden, conv2, KERNEL), nn.ReLU(),
            nn.ConvTranspose1d(conv2, conv1, KERNEL), nn.ReLU(),
        )
        self.out = nn.Linear(conv1, 1)

    def forward(self, h):
        z = h.transpose(0, 1).expand(-1, self.steps, -1)
        seq, _ = self.gru(z.contiguous(), h)
        y = self.deconv(seq.transpose(1, 2))
        return self.out(y.transpose(1, 2)).squeeze(-1)


class Extractor(nn.Module):
    def __init__(self, m, conv1=128, conv2=64, hidden=32):
        super().__init__()
        self.m = m
        self.encoder = Encoder(conv1, conv2, hidden)
        self.decoder = Decoder(m, conv1, conv2, hidden)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def rmse_loss(y_pred, y_true):
    return torch.sqrt(((y_pred - y_true) ** 2).mean(dim=1) + 1e-12).mean()
