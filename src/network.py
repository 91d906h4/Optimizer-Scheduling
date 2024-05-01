import torch

from torch import nn


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1      = self.conv(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2      = self.conv(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3      = self.conv(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4      = self.conv(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.linear1    = nn.Linear(1024, 100)

        self.relu       = nn.ReLU(inplace=True)
        self.flatten    = nn.Flatten()
        self.sigmoid    = nn.Sigmoid()

    @staticmethod
    def conv(in_channels: int, out_channels: int, kernel_size: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)

        x = self.linear1(x)

        x = self.sigmoid(x)

        return x