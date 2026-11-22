"""Image augmentations for SimCLR, with an on/off switch for each one.

What is an augmentation?
    An augmentation is a random edit of an image, for example cutting out a part of it
    (crop), mirroring it (flip) or changing its colours. The edited image still shows the
    same object.

Why does SimCLR need augmentations?
    SimCLR takes ONE unlabelled image, makes TWO different random edits of it (two "views"),
    and trains the network so that both views get similar outputs, while views of
    DIFFERENT images get different outputs. To succeed, the network has to recognise the
    same object even though the two views look different, so it learns what objects look
    like without needing any labels. Which edits we use decides what the network learns to
    ignore (position, colour, sharpness), which is why our project tests which
    augmentations matter most.

What is in this file?
    - simclr_transform(): builds the random edits used during SimCLR pretraining.
      Each of the five edits can be switched on or off by name, which the ablation uses.
    - test_transform(): no random edits, used for test images and feature extraction.
    - supervised_train_transform(): light edits, used when training a classifier on labels.
"""
# torchvision is PyTorch's library for images. Its "transforms" module contains ready-made
# image edits (crop, flip, colour changes, ...). We import it under the short name T,
# so we can write T.RandomHorizontalFlip() instead of the full name.
from torchvision import transforms as T

# A colour image has three channels: red, green and blue. These are the average value (MEAN)
# and the spread (STD, standard deviation) of each channel over STL-10 images, on a 0-1 scale.
# "Normalising" means: for every pixel, subtract the mean and divide by the standard
# deviation. The values then sit around 0 with a spread of about 1, which makes neural
# networks train more smoothly. We use the same numbers for every experiment.
MEAN, STD = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)

# The names of the five augmentations we study. A Python list in square brackets.
# Passing a shorter list to simclr_transform() switches the missing ones off.
ALL_AUGS = ["crop", "flip", "color", "gray", "blur"]


def simclr_transform(augs=ALL_AUGS, size=96, strength=0.5):
    """Build the chain of random edits that creates ONE view of an image.

    It is called twice on the same image (see TwoViews in src/data.py). Because every
    edit is random, the two results are different, and together they form a
    "positive pair" for the contrastive loss.

    Arguments (the values after "=" are the defaults used when nothing is given):
    augs     : list of augmentation names to switch ON. Anything not listed is OFF.
               The default is all five augmentations.
    size     : output size in pixels. STL-10 images are 96 x 96, so 96.
    strength : how strong the colour changes are. The SimCLR paper uses 1.0 for large
               ImageNet images and 0.5 for small CIFAR-10 images; STL-10 images are small,
               so we use 0.5.
    """
    # Turn the list into a "set". A set makes it quick and simple to check
    # whether a name is in it, e.g.  "crop" in augs  -> True or False.
    augs = set(augs)

    # An empty list. We will add the edits to it one by one, in the order they should run.
    ops = []

    # ---- 1. Random resized crop ----
    # T.RandomResizedCrop picks a random rectangle covering between 20% and 100% of the
    # image area (scale=(0.2, 1.0)), cuts it out, and resizes it back to size x size.
    # Two crops of the same image can show different parts of the object (for example the
    # head of a dog in one view and its body in the other), so the network must learn
    # features that connect the parts of an object.
    # The line uses Python's "A if condition else B" form:
    #   if "crop" is switched on  -> add the random crop,
    #   otherwise                 -> just resize to 96 x 96 so every image keeps the same shape.
    ops.append(T.RandomResizedCrop(size, scale=(0.2, 1.0)) if "crop" in augs else T.Resize((size, size)))

    # ---- 2. Horizontal flip ----
    # Mirror the image left to right. T.RandomHorizontalFlip does this with a 50% chance
    # (its default), so about half of the views are flipped.
    if "flip" in augs:
        ops.append(T.RandomHorizontalFlip())

    # ---- 3. Colour jitter ----
    # T.ColorJitter randomly changes four things. With strength = 0.5:
    #   brightness by up to 0.8 * 0.5 = 0.4  (40% brighter or darker)
    #   contrast   by up to 0.4
    #   saturation by up to 0.4 (how vivid the colours are)
    #   hue        by up to 0.2 * 0.5 = 0.1  (shifts the colours themselves)
    # T.RandomApply([...], p=0.8) applies this to only 80% of the views; the other 20% keep
    # their original colours.
    # Why: two crops of one image usually have the same colours, so without colour changes
    # the network could match the two views just by comparing colours, and learn little
    # about the actual object.
    if "color" in augs:
        ops.append(T.RandomApply([T.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)], p=0.8))

    # ---- 4. Random grayscale ----
    # Turn the image black and white with a 20% chance (p=0.2). Another way of stopping the
    # network from relying only on colour.
    if "gray" in augs:
        ops.append(T.RandomGrayscale(p=0.2))

    # ---- 5. Gaussian blur ----
    # Make the image blurry. 9 is the kernel size: each pixel is mixed with its neighbours in
    # a 9 x 9 window (about 10% of the 96-pixel width, the ratio the SimCLR paper uses).
    # sigma controls how strong the blur is; a random value between 0.1 (almost no blur) and
    # 2.0 (strong blur) is picked each time. It is applied to 50% of the views (p=0.5).
    if "blur" in augs:
        ops.append(T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5))

    # ---- Always done last ----
    # T.ToTensor() converts the image into a PyTorch tensor (a grid of numbers) with values
    # between 0 and 1, the format the network expects.
    # T.Normalize(MEAN, STD) then applies the normalisation explained at the top of the file.
    # "+=" adds both steps to the end of the list.
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]

    # T.Compose chains all the steps in the list into one function: give it an image, and it
    # runs every step in order and returns the final tensor.
    return T.Compose(ops)


def _finish(normalize):
    """The last steps shared by the two evaluation transforms below.

    Always converts the image to a tensor. Also normalises it, unless normalize is False.
    normalize=False is only used for an optional public checkpoint (from the sthalles/SimCLR
    project), which was trained on images without normalisation, so it must be given the
    same kind of input. The leading underscore in the name is a Python convention meaning
    "helper used only inside this file".
    """
    # "A if condition else B" again: two steps when normalising, otherwise only ToTensor.
    return [T.ToTensor(), T.Normalize(MEAN, STD)] if normalize else [T.ToTensor()]


def test_transform(normalize=True):
    """Transform for the test images and for extracting features. It has NO random edits.

    If the test images were randomly edited, the measured accuracy would change slightly
    every time we ran the test. Without random edits the result is the same every time,
    so different methods can be compared fairly.
    """
    return T.Compose(_finish(normalize))


def supervised_train_transform(size=96, normalize=True):
    """Light random edits used when training a classifier on the labelled images
    (the supervised baseline and fine-tuning in evaluate.py).

    Small random changes make the few labelled images look a little different each epoch,
    which reduces overfitting (memorising the training images). We use the same light
    edits for the baseline and for fine-tuning, so the only real difference between them
    is how their weights start.
    """
    # T.RandomCrop(size, padding=8): add an 8-pixel border around the image, then cut out a
    # random 96 x 96 window. The result is the image shifted by up to 8 pixels.
    # T.RandomHorizontalFlip(): mirror with a 50% chance.
    # *_finish(normalize): the "*" unpacks the list returned by _finish, so its steps
    # (ToTensor and maybe Normalize) are added to this list one by one.
    return T.Compose([T.RandomCrop(size, padding=8), T.RandomHorizontalFlip(), *_finish(normalize)])
