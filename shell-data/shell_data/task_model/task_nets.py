import torch.nn as nn
import logging
import torch


def _reset_parameters(m: nn.Module):
    """
    helper function to reset parameters of a model
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GroupNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    elif isinstance(m, nn.RNN):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    elif isinstance(m, nn.GRUCell):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    elif isinstance(m, nn.RNNCell):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)


class CIFAR10Net(nn.Module):
    def __init__(self, n_out=10) -> None:
        nn.Module.__init__(self)
        self.n_out = n_out
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.model = nn.Sequential(
            nn.Linear(4 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_out),
        )
        logging.info(
            f"CIFAR10 num parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

    def reset_parameters(self):
        self.apply(_reset_parameters)


class MNISTNet(nn.Module):
    def __init__(self, n_out=10) -> None:
        nn.Module.__init__(self)
        self.n_out = n_out
        self.embedding = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.model = nn.Sequential(
            nn.Linear(4 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_out),
        )

        logging.info(
            f"MNIST num parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.view(-1, 4 * 7 * 7)
        x = self.model(x)
        return x

    def reset_parameters(self):
        self.apply(_reset_parameters)
