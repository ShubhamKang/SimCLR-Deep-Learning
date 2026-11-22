"""Network definitions (report Section 3.1).

SimCLR has two parts:
  1. Encoder f (ResNet-18): turns a 96x96 image into a 512-number feature vector h.
     This is the part we KEEP after pretraining and reuse for classification.
  2. Projection head g: a small MLP that maps h (512 numbers) to z (128 numbers).
     The contrastive loss is computed on z. It is thrown away after pretraining.

QUESTIONS THIS FILE ANSWERS
Q: Why ResNet-18?
   A standard CNN for small and medium datasets (about 11 million parameters) that trains
   on one T4 GPU in our budget (50 epochs on 100,000 images took 3.1 hours). The same
   architecture is used for every method, so comparisons are fair.
Q: Why no ImageNet-pretrained weights?
   Starting from ImageNet weights would mean the network had already learned from millions
   of LABELLED images, which would defeat the purpose of testing what self-supervised
   pretraining on unlabelled STL-10 images alone can do.
Q: Why a projection head, and why discard it?
   Chen et al. found that computing the loss on a separate head and then using the features
   BEFORE the head gives better downstream accuracy. The head learns to throw away
   information that is not needed to match two views (e.g. colour), information that can
   still be useful for classification.
Q: What does the classifier look like?
   For evaluation (evaluate.py) a single linear layer 512 -> 10 is put on top of the encoder.
"""
import torch.nn as nn
import torchvision


def make_encoder():
    """ResNet-18 with RANDOM weights and its final classification layer removed.

    torchvision's ResNet-18 ends with `fc`, a layer that outputs 1000 ImageNet class scores.
    Replacing it with Identity (a layer that does nothing) makes the network output the
    512-dimensional pooled feature vector instead.
    """
    m = torchvision.models.resnet18(weights=None)  # weights=None -> random initialisation
    m.fc = nn.Identity()
    return m  # output shape: [batch, 512]


class SimCLR(nn.Module):
    """Encoder + projection head. Used only during pretraining (train_simclr.py).
    Only `self.encoder` is saved to the checkpoint used later for evaluation."""
    def __init__(self, proj_dim=128):
        super().__init__()
        self.encoder = make_encoder()
        # Projection head: 512 -> 512 -> 128 with a ReLU in between. The non-linearity
        # matters: SimCLR found a non-linear head better than a linear one or no head.
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))

    def forward(self, x):
        # image -> encoder -> 512-d features h -> head -> 128-d projection z (fed to the loss)
        return self.head(self.encoder(x))
