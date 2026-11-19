"""Network definitions.

SimCLR has two parts:
  1. Encoder  : a ResNet-18 that turns an image into a 512-number feature vector.
                This is the part we KEEP after pretraining and reuse for classification.
  2. Projection head : a small MLP that maps the 512 features to 128 numbers.
                The contrastive loss is computed on THESE numbers. The SimCLR paper
                found that the features before the head are better for downstream
                tasks, so the head is thrown away after pretraining.
"""
import torch.nn as nn
import torchvision


def make_encoder():
    """ResNet-18 with random weights (no ImageNet pretraining, which would defeat
    the purpose) and its final classification layer removed.

    Replacing `fc` with Identity makes the network output the 512-d pooled feature
    vector instead of 1000 ImageNet class scores.
    """
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Identity()
    return m  # output shape: [batch, 512]


class SimCLR(nn.Module):
    """Encoder + projection head, used only during pretraining."""
    def __init__(self, proj_dim=128):
        super().__init__()
        self.encoder = make_encoder()
        # Two-layer MLP with a ReLU in between (non-linear, as in the SimCLR paper).
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))

    def forward(self, x):
        # image -> 512-d features -> 128-d projection used by the loss
        return self.head(self.encoder(x))
