import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from PaSTiLa import pastila
import numpy as np


class Model(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, m: int):
        super().__init__()
        self.c1 = nn.Conv1d(input_shape, 128, 3, padding=1)
        self.c2 = nn.Conv1d(hidden_units, 64, 3, padding=1)
        self.g1 = nn.GRU(m, m)
        self.g2 = nn.GRU(m, m)
        self.t1 = nn.ConvTranspose1d(64, 64, 3, padding=1)
        self.t2 = nn.ConvTranspose1d(64, 128, 3, padding=1)
        self.t3 = nn.ConvTranspose1d(128, 128, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.c1(x)
        x = self.act(x)
        x = self.c2(x)
        x = self.act(x)
        x = self.t1(x)
        x = self.act(x)
        x = self.t2(x)
        x = self.act(x)
        x = self.t3(x)
        x = self.act(x)
        return x


def train_step(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    train_loss = 0
    model.train()
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)


def predict(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
):
    X_list, prediction_list, true_list = [], [], []
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            X_list.extend(X.cpu())
            prediction_list.extend(test_pred.cpu())
            true_list.extend(y.cpu())
        test_loss /= len(data_loader)
        print(f"Test loss: {test_loss:.5f}")
    return X_list, prediction_list, true_list


def prepare_data(input_filename, min_m, max_m, K, step=1, batch_size=32):
    best_snp = pastila(input_filename, min_m, max_m, K, step)
    m = len(best_snp[0][0])
    ts = np.loadtxt(input_filename)

    labels = np.zeros_like(ts).astype(int)
    for label, start, end in best_snp:
        labels[start:end] = label

    X = torch.tensor([ts[i : i + m] for i in range(len(ts) - m)], dtype=torch.float32)
    X = X.unsqueeze(dim=1)
    y = torch.tensor(
        [best_snp[0][int(labels[i])] for i in range(len(ts) - m)],
        dtype=torch.float32,
    )

    dataset = TensorDataset(X, y)

    train_set, test_set = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    return train_dataloader, test_dataloader
