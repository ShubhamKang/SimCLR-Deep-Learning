import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import STL10


class TwoViews(Dataset):
    """Returns two independently augmented views of the same image."""
    def __init__(self, base, transform):
        self.base, self.t = base, transform
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, _ = self.base[i]
        return self.t(img), self.t(img)


def unlabeled_set(root, transform, subset=None, seed=0):
    base = STL10(root, split="unlabeled", download=True)
    ds = TwoViews(base, transform)
    if subset:
        idx = np.random.RandomState(seed).permutation(len(ds))[:subset]
        ds = Subset(ds, idx.tolist())
    return ds


def labeled_set(root, split, transform, frac=1.0, seed=0):
    """split: 'train' (5k) or 'test' (8k). frac<1 gives a class-balanced subset of train."""
    ds = STL10(root, split=split, download=True, transform=transform)
    if frac >= 1.0 or split != "train":
        return ds
    labels = np.array(ds.labels)
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        idx += rng.permutation(ci)[: max(1, int(round(frac * len(ci))))].tolist()
    return Subset(ds, sorted(idx))
