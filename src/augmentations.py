"""SimCLR augmentations with per-augmentation on/off flags (used for the ablation)."""
from torchvision import transforms as T

MEAN, STD = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)
ALL_AUGS = ["crop", "flip", "color", "gray", "blur"]


def simclr_transform(augs=ALL_AUGS, size=96, strength=0.5):
    """augs: iterable of names from ALL_AUGS. Missing ones are switched off."""
    augs = set(augs)
    ops = []
    ops.append(T.RandomResizedCrop(size, scale=(0.2, 1.0)) if "crop" in augs else T.Resize((size, size)))
    if "flip" in augs:
        ops.append(T.RandomHorizontalFlip())
    if "color" in augs:
        ops.append(T.RandomApply([T.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)], p=0.8))
    if "gray" in augs:
        ops.append(T.RandomGrayscale(p=0.2))
    if "blur" in augs:
        ops.append(T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5))
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


def test_transform():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def supervised_train_transform(size=96):
    return T.Compose([T.RandomCrop(size, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
