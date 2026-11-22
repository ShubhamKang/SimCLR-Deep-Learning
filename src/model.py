"""The neural networks used in this project.

A neural network is a function with millions of adjustable numbers ("weights"). Training
changes these weights so the outputs become useful.

SimCLR uses two parts:
  1. The encoder: a ResNet-18, a well-known convolutional neural network (CNN) for images,
     with 18 layers and about 11 million weights. It turns a 96 x 96 image into a list of
     512 numbers (a "feature vector") that describes the image. This is the part we keep
     after pretraining and use later for classification.
  2. The projection head: a small extra network that turns those 512 numbers into 128
     numbers. The contrastive loss is calculated on these 128 numbers. After pretraining
     the head is thrown away, because the SimCLR paper found that the 512 numbers BEFORE
     the head work better for classifying images.
"""
# torch.nn (short name nn) contains the building blocks of neural networks:
# layers such as nn.Linear, activation functions such as nn.ReLU, and nn.Module.
import torch.nn as nn
# torchvision contains ready-made image networks, including ResNet-18.
import torchvision


def make_encoder():
    """Create a ResNet-18 encoder that outputs 512 numbers per image.

    The weights start random. We deliberately do NOT load weights pretrained on ImageNet
    (a large labelled dataset), because the point of the project is to see what the
    network can learn from unlabelled STL-10 images alone.
    """
    # Build a ResNet-18. weights=None means random starting weights.
    m = torchvision.models.resnet18(weights=None)
    # The last layer of ResNet-18 is called "fc" (fully connected). It turns the 512 features
    # into 1,000 scores for the 1,000 ImageNet classes, which we do not need.
    # nn.Identity() is a layer that returns its input unchanged, so replacing fc with it
    # makes the network output the 512 features directly.
    m.fc = nn.Identity()
    # For a batch of images the output has the shape [number of images, 512].
    return m


class SimCLR(nn.Module):
    """Encoder plus projection head. Only used during SimCLR pretraining (train_simclr.py).

    nn.Module is PyTorch's base class for networks. A network class defines its layers in
    __init__ and what happens to the input in forward.
    """

    def __init__(self, proj_dim=128):
        # Run the setup code of nn.Module first; PyTorch requires this.
        super().__init__()
        # The encoder: ResNet-18 that outputs 512 numbers (see make_encoder above).
        self.encoder = make_encoder()
        # The projection head, built with nn.Sequential, which runs layers one after another:
        #   nn.Linear(512, 512)      : a fully connected layer, 512 numbers in, 512 out.
        #   nn.ReLU(inplace=True)    : an activation function that replaces negative numbers
        #                              with 0. Without it, two Linear layers in a row would be
        #                              no more powerful than one. inplace=True saves memory.
        #   nn.Linear(512, proj_dim) : 512 numbers in, 128 out (proj_dim = 128).
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))

    def forward(self, x):
        # forward says what the network does with an input x (a batch of images).
        # First the encoder turns each image into 512 numbers, then the head turns those
        # into 128 numbers. Calling model(x) runs this function.
        return self.head(self.encoder(x))
