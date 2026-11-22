"""Data loading for STL-10.

STL-10 (Coates et al., 2011) has three parts that we use:
  - "unlabeled": 100,000 images, no labels       -> SimCLR pretraining only
  - "train"    : 5,000 labelled images (10 classes x 500) -> training the classifiers
  - "test"     : 8,000 labelled images (800 per class)    -> every reported accuracy
All images are 96x96 colour. The 10 classes are airplane, bird, car, cat, deer, dog,
horse, monkey, ship, truck.

QUESTIONS THIS FILE ANSWERS
Q: Why STL-10?
   It is built for self-supervised / semi-supervised learning: many unlabelled images,
   few labels. Its unlabelled images also include objects outside the 10 classes, which
   makes pretraining more realistic.
Q: How did you simulate label scarcity?
   labeled_set(frac=...) keeps 100%, 20%, 10% or 1% of the training labels, which is
   500, 100, 50 or 5 images per class (5,000 / 1,000 / 500 / 50 images in total).
Q: How do you make sure every method sees the same labelled images?
   The subset is chosen with a fixed random seed (seed 0), so the supervised baseline,
   the linear probe and fine-tuning all train on exactly the same images at each fraction.
Q: Why class-balanced subsets?
   We take the same share from every class, so even the 1% subset has 5 images of every
   class and cannot accidentally miss one.
Q: Why does the ablation use only 20,000 unlabelled images?
   Compute budget: unlabeled_set(subset=20000) picks a fixed random 20,000 images (same
   seed for every run), so all ablation runs see the SAME images and stay comparable.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import STL10


class TwoViews(Dataset):
    """Wraps a dataset so that each item returns TWO different augmented views of the
    same image.

    The transform contains random operations, so calling it twice on the same image gives
    two different results. Those two views form a "positive pair" in the contrastive loss.
    Labels are ignored on purpose: this is self-supervised learning.
    """
    def __init__(self, base, transform):
        self.base, self.t = base, transform  # base = the raw STL-10 split, t = simclr_transform

    def __len__(self):
        return len(self.base)  # number of images (100,000, or the subset size)

    def __getitem__(self, i):
        img, _ = self.base[i]            # the label (unlabelled split has none) is thrown away
        return self.t(img), self.t(img)  # two independent random augmentations = one positive pair


def unlabeled_set(root, transform, subset=None, seed=0):
    """The unlabelled images, returned as pairs of views, for SimCLR pretraining.

    root   : folder where STL-10 is stored (downloaded automatically, about 2.6 GB).
    subset : if given, use only this many images. The ablations use 20,000 so that each
             run takes minutes instead of hours.
    seed   : makes the random subset reproducible, so every ablation run sees the SAME
             20,000 images and differences come only from the setting being tested.
    """
    base = STL10(root, split="unlabeled", download=True)  # download=True fetches it once
    ds = TwoViews(base, transform)
    if subset:
        # Shuffle all indices with a fixed seed and keep the first `subset` of them.
        idx = np.random.RandomState(seed).permutation(len(ds))[:subset]
        ds = Subset(ds, idx.tolist())
    return ds


def labeled_set(root, split, transform, frac=1.0, seed=0):
    """Labelled images for training a classifier or for testing.

    split : "train" (5,000 images) or "test" (8,000 images).
    frac  : fraction of the training labels to use: 1.0, 0.2, 0.1 or 0.01 in the report
            (500, 100, 50 or 5 images per class). This simulates label scarcity.
            The test set is never reduced, so accuracies are always on all 8,000 images.
    seed  : same seed -> same chosen images, so the supervised baseline and the SimCLR
            models are compared on exactly the same few labels.

    The subset is CLASS-BALANCED: the same share is taken from every class.
    """
    ds = STL10(root, split=split, download=True, transform=transform)
    if frac >= 1.0 or split != "train":
        return ds  # full training set, or the test set (never subsampled)
    labels = np.array(ds.labels)          # class index (0-9) of every training image
    rng = np.random.RandomState(seed)     # seeded generator -> reproducible choice
    idx = []
    for c in np.unique(labels):           # go class by class -> class-balanced subset
        class_idx = np.where(labels == c)[0]                      # all images of class c (500)
        n_keep = max(1, int(round(frac * len(class_idx))))        # e.g. 0.01 * 500 = 5; keep at least one
        idx += rng.permutation(class_idx)[:n_keep].tolist()       # random choice within the class
    return Subset(ds, sorted(idx))
