"""Image augmentations for SimCLR, with an on/off switch for each one.

WHAT THIS FILE DOES
SimCLR learns by taking ONE unlabelled image, making TWO random augmented versions
("views") of it, and training the network so both views get similar features while
views of DIFFERENT images get different features. The augmentations decide what the
network is forced to ignore (position, colour, sharpness...), so they decide what it
learns. This is why "which augmentations matter most?" is our main research question
(report Section 6), and why every augmentation here can be switched on or off by name.

WHERE IT IS USED
- train_simclr.py builds the pretraining views with simclr_transform(augs).
- The augmentation ablation (scripts/run_ablation.py, report Table 3) passes a different
  `augs` list per run: e.g. ["flip", "color", "gray", "blur"] is the "no_crop" setting.
- evaluate.py uses test_transform() and supervised_train_transform().

QUESTIONS THIS FILE ANSWERS (with the numbers from the report)
Q: Which augmentations did you use and with what settings?
   Random resized crop (scale 0.2-1.0), horizontal flip (p=0.5), colour jitter
   (brightness/contrast/saturation 0.4, hue 0.1, p=0.8), grayscale (p=0.2),
   Gaussian blur (kernel 9, sigma 0.1-2.0, p=0.5). See the code below.
Q: Why is crop the most important? (no_crop 38.2% vs full 57.0%, none 36.9%)
   Two random crops often show DIFFERENT parts of the object, so to match them the
   network must learn shape/texture features that connect the parts. Without crop the
   two views show the same full image and the task becomes too easy.
Q: Why does colour jitter matter? (no_color 50.0%, -7.0 points; crop+colour 53.0% vs crop 45.1%)
   Two crops of one image share the same colours, so without colour changes the network
   can match views by colour statistics alone (a shortcut) and learns little about content.
Q: Why did blur not help? (no_blur 58.6%, +1.6 points)
   96x96 images already have little detail, so blur can remove useful information, and a
   20-epoch schedule may be too short to benefit from weak augmentations. The difference
   is within run-to-run variation, so the report calls it "no measurable benefit".
Q: Why is the test set not augmented?
   Accuracy must be deterministic and comparable between runs (see test_transform).
"""
from torchvision import transforms as T

# Per-channel (red, green, blue) mean and standard deviation of STL-10 pixel values.
# Normalising with them centres the inputs around 0 with spread about 1, which makes
# training more stable. The same values are used for every run, so they do not affect
# comparisons between methods.
MEAN, STD = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)

# Names of the five augmentations we study. Any subset can be passed to simclr_transform().
# These names are what the ablation settings switch on and off (report Table 3).
ALL_AUGS = ["crop", "flip", "color", "gray", "blur"]


def simclr_transform(augs=ALL_AUGS, size=96, strength=0.5):
    """Build the augmentation pipeline that creates ONE view of an image.

    It is applied twice to the same image (see TwoViews in src/data.py); because every
    step is random, the two results differ and form a "positive pair".

    augs     : names of augmentations that are ON. Anything not listed is OFF.
               Default = all five (the "full" setting, 57.0% in the ablation).
    size     : output size in pixels. STL-10 images are 96x96.
    strength : colour-jitter strength. The SimCLR paper uses 1.0 for ImageNet and 0.5 for
               the small CIFAR-10 images; we use 0.5 because STL-10 images are also small.

    Order of operations: geometry first (crop, flip), then colour (jitter, grayscale),
    then blur, then conversion to a tensor. This follows the SimCLR recipe.
    """
    augs = set(augs)  # a set makes the "is it on?" checks below simple
    ops = []          # list of operations, applied in order

    # RANDOM RESIZED CROP (the most important augmentation in our results).
    # Picks a random rectangle covering 20%-100% of the image area (random aspect ratio)
    # and resizes it back to `size`. Two crops of the same image may show different parts,
    # e.g. a dog's head in one view and its body in the other, so the network must learn
    # features that link parts of an object together.
    # We use a minimum scale of 0.2 because at 96x96, very small crops would contain only a
    # few pixels of the object.
    # If crop is OFF we only resize, so every image still has the same 96x96 shape.
    ops.append(T.RandomResizedCrop(size, scale=(0.2, 1.0)) if "crop" in augs else T.Resize((size, size)))

    # HORIZONTAL FLIP: mirror the image left-right with 50% probability (torchvision default).
    # Cheap and mild; removing it changed accuracy by only 0.1 points (no_flip 56.9%).
    if "flip" in augs:
        ops.append(T.RandomHorizontalFlip())

    # COLOUR JITTER: randomly change brightness, contrast and saturation by up to
    # 0.8*strength = 0.4, and hue by up to 0.2*strength = 0.1. Applied to 80% of images
    # (RandomApply p=0.8). It removes the "match the views by colour" shortcut.
    if "color" in augs:
        ops.append(T.RandomApply([T.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)], p=0.8))

    # RANDOM GRAYSCALE: convert 20% of images to black and white, another colour distortion
    # so colour is not a reliable cue. Removing it cost 5.8 points (no_gray 51.2%).
    if "gray" in augs:
        ops.append(T.RandomGrayscale(p=0.2))

    # GAUSSIAN BLUR on 50% of images. Kernel size 9 is about 10% of the 96-pixel width,
    # the ratio the SimCLR paper uses; sigma (blur strength) is sampled between 0.1 and 2.0.
    # In our 20-epoch ablation it gave no benefit (no_blur 58.6% vs full 57.0%) and it is
    # one of the slower augmentations on the CPU.
    if "blur" in augs:
        ops.append(T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5))

    # Always last: PIL image -> float tensor with values in [0, 1], then normalise.
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(ops)  # Compose chains the operations into one callable


def _finish(normalize):
    """Final steps shared by the evaluation transforms: ToTensor, plus mean/std
    normalisation unless normalize=False.

    normalize=False is only used for the optional public checkpoint (sthalles/SimCLR),
    which was trained on images WITHOUT normalisation, so it must see the same inputs.
    """
    return [T.ToTensor(), T.Normalize(MEAN, STD)] if normalize else [T.ToTensor()]


def test_transform(normalize=True):
    """No randomness at all. Used for the 8,000 test images and for extracting linear-probe
    features, so accuracy numbers are deterministic and comparable between runs.
    Augmenting the test set would make the measured accuracy change from run to run."""
    return T.Compose(_finish(normalize))


def supervised_train_transform(size=96, normalize=True):
    """Light augmentation for the supervised baseline and for fine-tuning.

    Pad the image by 8 pixels on each side and take a random 96x96 crop (a small random
    shift), plus a horizontal flip. This reduces overfitting on the few labelled images.

    Q: Why the same transform for the baseline and fine-tuning?
       So the only real difference between them is the starting weights (random vs SimCLR),
       which makes the comparison in report Table 1 fair.
    Q: Why not the strong SimCLR augmentations here?
       Strong augmentation is needed to make the contrastive task hard; for supervised
       training on labels a light, standard augmentation is enough.
    """
    return T.Compose([T.RandomCrop(size, padding=8), T.RandomHorizontalFlip(), *_finish(normalize)])
