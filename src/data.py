"""Data loading for STL-10.

STL-10 has three parts that we use:
  - "unlabeled": 100,000 images with no labels   -> used for SimCLR pretraining
  - "train"    : 5,000 labelled images (10 classes x 500) -> few-label training
  - "test"     : 8,000 labelled images           -> final accuracy is always measured here
"""
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import STL10


class TwoViews(Dataset):
    """Wraps a dataset so that each item returns TWO different augmented views
    of the same image.

    The transform contains random operations, so calling it twice on the same image
    gives two different results. The two views form a "positive pair" in the
    contrastive loss. Labels are ignored on purpose (self-supervised learning).
    """
    def __init__(self, base, transform):
        self.base, self.t = base, transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]            # the label (if any) is thrown away
        return self.t(img), self.t(img)  # two independent random augmentations


def unlabeled_set(root, transform, subset=None, seed=0):
    """The unlabelled images, returned as pairs of views, for SimCLR pretraining.

    subset : if given, use only this many random images. The ablation runs use a
             smaller subset (e.g. 20,000) to finish quickly on a small compute budget.
    seed   : makes the random subset reproducible, so every ablation run sees the
             SAME images and the comparison between them is fair.
    """
    base = STL10(root, split="unlabeled", download=True)  # download=True fetches it once
    ds = TwoViews(base, transform)
    if subset:
        idx = np.random.RandomState(seed).permutation(len(ds))[:subset]
        ds = Subset(ds, idx.tolist())
    return ds


def labeled_set(root, split, transform, frac=1.0, seed=0):
    """Labelled images for training a classifier or for testing.

    split : "train" (5k images) or "test" (8k images).
    frac  : fraction of the training labels to use, e.g. 0.1 = 10% = 50 images per class.
            This simulates label scarcity. The test set is never reduced.
    seed  : same seed -> same chosen images, so supervised and SimCLR models
            are compared on exactly the same few labels.

    The subset is CLASS-BALANCED: we take the same share from every class, so a
    small fraction cannot accidentally miss a class.
    """
    ds = STL10(root, split=split, download=True, transform=transform)
    if frac >= 1.0 or split != "train":
        return ds
    labels = np.array(ds.labels)
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(labels):
        class_idx = np.where(labels == c)[0]                      # all images of class c
        n_keep = max(1, int(round(frac * len(class_idx))))        # keep at least one
        idx += rng.permutation(class_idx)[:n_keep].tolist()       # random choice within the class
    return Subset(ds, sorted(idx))
