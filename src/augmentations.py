"""Image augmentations for SimCLR, with a switch for each one.

Why this file matters:
SimCLR learns by taking ONE image, making TWO random augmented versions ("views"),
and training the network so both views get similar features while views of
DIFFERENT images get different features. The augmentations decide what the network
is forced to ignore (e.g. colour, position), so they strongly shape what it learns.

Our research question is "which augmentations matter most?", so every augmentation
can be turned on or off by name. The ablation scripts use this to train models
with one augmentation removed, or with only a few kept.
"""
from torchvision import transforms as T

# Per-channel mean and std of STL-10 images. Normalising inputs with these makes
# training more stable (values centred around 0 with unit-ish spread).
MEAN, STD = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)

# Names of the five augmentations we study. Pass any subset to simclr_transform().
ALL_AUGS = ["crop", "flip", "color", "gray", "blur"]


def simclr_transform(augs=ALL_AUGS, size=96, strength=0.5):
    """Build the augmentation pipeline used to create ONE view of an image.

    augs     : names of augmentations that are ON. Anything not listed is OFF.
    size     : output image size in pixels (STL-10 images are 96x96).
    strength : colour-jitter strength. SimCLR uses 1.0 on ImageNet; 0.5 is gentler
               and suits small 96x96 images.

    The order of operations below is: geometry first (crop, flip), then colour
    (jitter, grayscale), then blur, and finally conversion to a tensor.
    """
    augs = set(augs)
    ops = []

    # Random resized crop: take a random patch covering 20%-100% of the image and
    # resize it back to `size`. This is the most important SimCLR augmentation:
    # two crops of the same image often show different parts, so the network must
    # learn features that link parts of an object together.
    # If crop is OFF, we just resize so that the image shape stays the same.
    ops.append(T.RandomResizedCrop(size, scale=(0.2, 1.0)) if "crop" in augs else T.Resize((size, size)))

    # Horizontal flip with 50% chance (the default probability). Cheap and mild.
    if "flip" in augs:
        ops.append(T.RandomHorizontalFlip())

    # Colour jitter: randomly change brightness, contrast, saturation (0.8*strength)
    # and hue (0.2*strength). Applied to 80% of images. It stops the network from
    # "cheating" by matching the two views through their colour histogram alone.
    if "color" in augs:
        ops.append(T.RandomApply([T.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)], p=0.8))

    # Random grayscale: turn 20% of images to grayscale, so colour is not the only cue.
    if "gray" in augs:
        ops.append(T.RandomGrayscale(p=0.2))

    # Gaussian blur on 50% of images. Kernel 9 is about 10% of the image width,
    # the ratio the SimCLR paper uses. Sigma is sampled between 0.1 and 2.0.
    if "blur" in augs:
        ops.append(T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5))

    # Always: convert the PIL image to a float tensor in [0,1], then normalise.
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


def _finish(normalize):
    """ToTensor, plus mean/std normalisation unless normalize=False (public checkpoint)."""
    return [T.ToTensor(), T.Normalize(MEAN, STD)] if normalize else [T.ToTensor()]


def test_transform(normalize=True):
    """No randomness. Used for the test set and for extracting features, so that
    accuracy numbers are deterministic and comparable between runs."""
    return T.Compose(_finish(normalize))


def supervised_train_transform(size=96, normalize=True):
    """Light augmentation for supervised training / fine-tuning on the labelled data.

    Pad by 8 pixels then crop back (a small random shift) plus a horizontal flip.
    We use the SAME light recipe for the from-scratch baseline and for fine-tuning,
    so the comparison between them is fair.
    """
    return T.Compose([T.RandomCrop(size, padding=8), T.RandomHorizontalFlip(), *_finish(normalize)])
