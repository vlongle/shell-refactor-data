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


# class CIFAR10Net(nn.Module):
#     def __init__(self, n_out=10) -> None:
#         nn.Module.__init__(self)
#         self.n_out = n_out
#         self.embedding = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(16, 4, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#         )
#         self.model = nn.Sequential(
#             nn.Linear(4 * 8 * 8, 64),
#             nn.ReLU(),
#             nn.Linear(64, self.n_out),
#         )
#         logging.info(
#             f"CIFAR10 num parameters: {sum(p.numel() for p in self.parameters())}")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.embedding(x)
#         x = x.view(x.size(0), -1)
#         x = self.model(x)
#         return x

#     def reset_parameters(self):
#         self.apply(_reset_parameters)


# VGG16 Net

vgg16_arch = [64, 64, 'M', 128, 128, 'M', 256, 256,
              'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class CIFAR10Net(nn.Module):
    def __init__(self, n_out):
        self.in_channels = 3
        self.n_out = n_out
        super().__init__()
        self.conv_layers = self.create_conv_layers(vgg16_arch)
        self.fcs = nn.Sequential(
            nn.Linear(in_features=512*1*1, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.n_out)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels
        for x in arch:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU(),
                           ]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)


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


# class MNISTNet(nn.Module):
#     def __init__(self, n_out=10) -> None:
#         nn.Module.__init__(self)
#         self.n_out = n_out
#         self.model = nn.Sequential(
#             nn.Linear(28 * 28, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.n_out),
#         )

#         logging.info(
#             f"MNIST num parameters: {sum(p.numel() for p in self.parameters())}")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.view(-1, 28 * 28)
#         x = self.model(x)
#         return x

#     def reset_parameters(self):
#         self.apply(_reset_parameters)


class FashionMNISTNet(nn.Module):
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
            f"Fashion MNIST num parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.view(-1, 4 * 7 * 7)
        x = self.model(x)
        return x

    def reset_parameters(self):
        self.apply(_reset_parameters)
