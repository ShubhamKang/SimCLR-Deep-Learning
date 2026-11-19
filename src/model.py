import torch.nn as nn
import torchvision


def make_encoder():
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Identity()
    return m  # outputs 512-d features


class SimCLR(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.encoder = make_encoder()
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))
    def forward(self, x):
        return self.head(self.encoder(x))
