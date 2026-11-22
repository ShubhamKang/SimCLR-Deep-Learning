"""Loading the STL-10 dataset.

STL-10 is a public image dataset with three parts that we use:
  - "unlabeled": 100,000 images without labels -> used only for SimCLR pretraining
  - "train"    : 5,000 labelled images (10 classes, 500 each) -> used to train classifiers
  - "test"     : 8,000 labelled images (800 per class) -> used to measure accuracy
All images are 96 x 96 pixels, in colour. The 10 classes are airplane, bird, car, cat,
deer, dog, horse, monkey, ship and truck.

A "label" is the correct class of an image, written by a person. A "dataset" in PyTorch
is an object that can tell how many items it has and give back item number i.
"""
# numpy (short name np) is the standard Python library for working with arrays of numbers.
import numpy as np
# torch is PyTorch, the deep learning library used in this project.
import torch
# Dataset: the base class for custom datasets in PyTorch.
# Subset: wraps a dataset so that only some of its items (given by their positions) are used.
from torch.utils.data import Dataset, Subset
# STL10: torchvision's ready-made loader for the STL-10 dataset (it can also download it).
from torchvision.datasets import STL10


class TwoViews(Dataset):
    """A dataset wrapper that gives back TWO differently edited versions of each image.

    SimCLR needs two "views" of every image. The transform contains random edits, so
    applying it twice to the same image gives two different results. These two views
    are a "positive pair": the network is trained to give them similar outputs.
    """

    def __init__(self, base, transform):
        # __init__ runs once, when the object is created.
        # base      = the original STL-10 dataset (here the unlabelled part)
        # transform = the random edits from simclr_transform() in src/augmentations.py
        # self.base and self.t store them inside the object for later use.
        self.base, self.t = base, transform

    def __len__(self):
        # __len__ tells PyTorch how many items the dataset has: the same number as the
        # original dataset (100,000, or fewer if a subset is used).
        return len(self.base)

    def __getitem__(self, i):
        # __getitem__ gives back item number i. PyTorch calls it for each image in a batch.
        # self.base[i] returns (image, label). The unlabelled part has no real label, and
        # SimCLR does not use labels anyway, so we throw it away by naming it "_".
        img, _ = self.base[i]
        # Apply the random edits twice to the same image -> two different views.
        return self.t(img), self.t(img)


def unlabeled_set(root, transform, subset=None, seed=0):
    """Return the unlabelled images, as pairs of views, for SimCLR pretraining.

    root      : folder where STL-10 is saved (downloaded there the first time, about 2.6 GB).
    transform : the random edits used to make each view.
    subset    : if a number is given, only that many images are used. The ablation
                experiments use 20,000 images so each run takes minutes instead of hours.
                None (the default) means use all 100,000.
    seed      : starting value for the random choice of the subset. The same seed always
                picks the same images, so every ablation run trains on exactly the same
                20,000 images and the runs can be compared fairly.
    """
    # Load the unlabelled part of STL-10. download=True downloads it if it is not there yet.
    base = STL10(root, split="unlabeled", download=True)
    # Wrap it so that each item becomes a pair of views.
    ds = TwoViews(base, transform)
    # Only if a subset size was given:
    if subset:
        # np.random.RandomState(seed) makes a random number generator that always gives the
        # same sequence for the same seed. .permutation(len(ds)) shuffles the numbers
        # 0, 1, ..., 99,999 into a random order. [:subset] keeps the first `subset` of them.
        idx = np.random.RandomState(seed).permutation(len(ds))[:subset]
        # Keep only the images at those positions. .tolist() turns the numpy array into a
        # normal Python list, which Subset expects.
        ds = Subset(ds, idx.tolist())
    return ds


def labeled_set(root, split, transform, frac=1.0, seed=0):
    """Return labelled images, for training a classifier or for testing it.

    split     : "train" (the 5,000 training images) or "test" (the 8,000 test images).
    transform : the edits applied to each image.
    frac      : what fraction of the training labels to use. 1.0 = all, 0.2 = 20%,
                0.1 = 10%, 0.01 = 1%. This is how we pretend labels are scarce.
                With 500 images per class, 0.01 keeps 5 images per class (50 in total).
                The test set is never reduced.
    seed      : the same seed always picks the same training images, so every method
                (baseline, linear probe, fine-tuning) is trained on exactly the same images.

    The chosen images are "class-balanced": the same share is taken from every class,
    so even the 1% subset has images of all 10 classes.
    """
    # Load the requested part of STL-10, with the given edits applied to each image.
    ds = STL10(root, split=split, download=True, transform=transform)
    # If all labels are wanted, or this is the test set, return everything unchanged.
    if frac >= 1.0 or split != "train":
        return ds
    # ds.labels holds the class number (0 to 9) of every training image.
    labels = np.array(ds.labels)
    # A random number generator fixed by the seed, so the choice is repeatable.
    rng = np.random.RandomState(seed)
    # An empty list that will collect the positions of the images we keep.
    idx = []
    # np.unique(labels) gives each class number once: 0, 1, ..., 9. We go class by class.
    for c in np.unique(labels):
        # np.where(labels == c)[0] gives the positions of all images of class c (500 of them).
        class_idx = np.where(labels == c)[0]
        # How many to keep: the fraction times 500, rounded to a whole number,
        # but never fewer than 1. For frac=0.01: round(0.01 * 500) = 5.
        n_keep = max(1, int(round(frac * len(class_idx))))
        # Shuffle that class's positions randomly and keep the first n_keep of them.
        # "+=" adds them to the list of kept positions.
        idx += rng.permutation(class_idx)[:n_keep].tolist()
    # Return only the chosen images. sorted() just puts the positions in increasing order.
    return Subset(ds, sorted(idx))
