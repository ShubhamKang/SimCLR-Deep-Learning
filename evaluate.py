"""Measure test accuracy on STL-10 when only a few labels are available.

This script builds an image classifier in one of three ways (chosen with --mode):
  scratch  : a ResNet-18 that starts with random weights and is trained only on the
             labelled images. This is the "supervised baseline": what you get WITHOUT
             SimCLR pretraining.
  linear   : a "linear probe". Load the SimCLR encoder, freeze it (its weights are not
             changed), and train only one small layer on top that turns the 512 features
             into 10 class scores. This measures how useful the learned features are.
  finetune : load the SimCLR encoder and keep training ALL of its layers on the labelled
             images, with a small learning rate.

--frac chooses how many of the 5,000 training labels to use (1.0 = all, 0.2, 0.1, 0.01).
Accuracy is always measured on all 8,000 test images, and each run adds one line to
results/results.csv.

Example:
  python evaluate.py --mode linear --ckpt checkpoints/full.pt --frac 0.1 --tag simclr_full
"""
# argparse reads command-line options, csv writes the results file, os handles paths.
import argparse, csv, os
# torch is PyTorch; torch.nn (nn) contains layers and loss functions.
import torch, torch.nn as nn
# DataLoader serves a dataset in batches.
from torch.utils.data import DataLoader
# Our own code: image transforms, the labelled dataset loader, and the ResNet-18 encoder.
from src.augmentations import test_transform, supervised_train_transform
from src.data import labeled_set
from src.model import make_encoder

# ---------------- command-line options ----------------
p = argparse.ArgumentParser()
# Which of the three methods to run. choices=[...] rejects any other value.
p.add_argument("--mode", choices=["scratch", "linear", "finetune"], required=True)
# Path to the saved SimCLR encoder (not needed for scratch).
p.add_argument("--ckpt", default=None, help="SimCLR checkpoint (needed for linear and finetune)")
# Fraction of the training labels to use.
p.add_argument("--frac", type=float, default=1.0, help="fraction of training labels to use")
# A short name written into results.csv so make_plots.py can group the rows (e.g. simclr_full).
p.add_argument("--tag", default="", help="label for this run in results.csv, used by make_plots.py")
# Optional overrides. If not given (None), the defaults further down are used:
# 300 training steps for the linear probe, 100 epochs for scratch and finetune.
p.add_argument("--epochs", type=int, default=None, help="override default epochs")
p.add_argument("--lr", type=float, default=None, help="override default learning rate")
# Seed: decides which labelled images are picked and the random starting weights.
p.add_argument("--seed", type=int, default=0, help="changes which labelled images are picked and the init")
# Folder where STL-10 is stored.
p.add_argument("--data", default="data")
# Only for the optional public checkpoint, which was trained without normalised images.
p.add_argument("--no_normalize", action="store_true",
               help="skip mean/std normalisation (for the public sthalles checkpoint, trained without it)")
# Read the typed options into a.mode, a.frac, ...
a = p.parse_args()
# Fix the random seed so the run can be repeated exactly.
torch.manual_seed(a.seed)
# norm is True (normalise images) unless --no_normalize was given.
norm = not a.no_normalize
# Use the GPU if there is one, otherwise the CPU.
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- build the encoder ----------------
# Create a ResNet-18 encoder with random weights (outputs 512 numbers per image).
enc = make_encoder()
# For linear and finetune, replace the random weights with the SimCLR-pretrained ones.
if a.mode != "scratch":
    # torch.load reads the checkpoint file; ["encoder"] picks the saved encoder weights
    # (see the torch.save line in train_simclr.py). map_location="cpu" loads them on the CPU
    # first, which works whether or not a GPU is present. load_state_dict copies them in.
    enc.load_state_dict(torch.load(a.ckpt, map_location="cpu")["encoder"])
# Move the encoder to the GPU or CPU.
enc.to(dev)


def acc(model, dl):
    """Return the fraction of images in the data loader dl that the model classifies correctly."""
    # Evaluation mode: layers such as batch normalisation use their fixed, learned values.
    model.eval()
    # Counters for correct predictions and for the number of images seen.
    correct = n = 0
    # torch.no_grad(): we only measure, we do not train, so PyTorch does not need to keep
    # track of gradients. This saves memory and time.
    with torch.no_grad():
        # x = a batch of images, y = their true classes.
        for x, y in dl:
            # model(...) gives 10 scores per image; argmax(1) picks the class with the highest
            # score (the prediction). Compare with the true classes and count the matches.
            correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
            # Count the images in this batch.
            n += len(y)
    # Accuracy = correct predictions / all images.
    return correct / n


# The test data: all 8,000 test images, no random edits, in batches of 512.
# num_workers=4: four CPU processes load the images in parallel.
# The test set is the same for every experiment, so all accuracies can be compared directly.
test_dl = DataLoader(labeled_set(a.data, "test", test_transform(norm)), 512, num_workers=4)

if a.mode == "linear":
    # ================= linear probe =================
    # @torch.no_grad() means: no gradients inside this function (we only use the encoder).
    @torch.no_grad()
    def feats(dl):
        """Pass every image through the FROZEN encoder once and collect the 512 features
        of each image together with its label."""
        # Evaluation mode for the encoder.
        enc.eval()
        # Lists that collect the features (F) and labels (Y) batch by batch.
        F, Y = [], []
        for x, y in dl:
            # Features of this batch, moved back to the CPU to be collected.
            F.append(enc(x.to(dev)).cpu())
            Y.append(y)
        # torch.cat joins the per-batch pieces into one big table of features and one list
        # of labels.
        return torch.cat(F), torch.cat(Y)

    # Features and labels of the chosen training images (the --frac subset), and of the test
    # images. No random edits here: the encoder is frozen, so each image only needs to pass
    # through it once, which makes the linear probe very fast.
    ftr, ytr = feats(DataLoader(labeled_set(a.data, "train", test_transform(norm), a.frac, a.seed), 512, num_workers=4))
    fte, yte = feats(test_dl)

    # Standardise the features: for each of the 512 features, subtract its average and divide
    # by its spread, so all features are on a similar scale. This helps the linear layer train
    # quickly. The average (mu) and spread (sd) come from the TRAINING features only; using
    # the test features here would let information from the test set leak into training.
    # + 1e-6 avoids dividing by zero if a feature never changes.
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    # Apply the same standardisation to training and test features and move them to the GPU.
    ftr, fte = ((ftr - mu) / sd).to(dev), ((fte - mu) / sd).to(dev)
    ytr = ytr.to(dev)

    # The classifier: a single linear layer that turns 512 features into 10 class scores.
    # This is the ONLY thing trained in the linear probe.
    clf = nn.Linear(512, 10).to(dev)
    # Adam optimiser for this layer. "a.lr or 1e-3" means: use --lr if it was given,
    # otherwise 0.001. weight_decay=1e-4 discourages very large weights (less overfitting).
    opt = torch.optim.Adam(clf.parameters(), lr=a.lr or 1e-3, weight_decay=1e-4)
    # 300 training steps (unless --epochs was given). Each step uses ALL training features at
    # once ("full-batch"), which is possible because they are small (at most 5,000 x 512).
    for _ in range(a.epochs or 300):
        # Cross-entropy loss between the predicted scores and the true labels.
        loss = nn.functional.cross_entropy(clf(ftr), ytr)
        # Clear old gradients, compute new ones, update the layer's weights.
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Test accuracy: predicted class (argmax) compared with the true class, as a fraction.
    # .float().mean() turns True/False into 1/0 and averages them; .item() gives a Python number.
    result = (clf(fte).argmax(1).cpu() == yte).float().mean().item()

else:
    # ================= scratch baseline and fine-tuning =================
    # Both use the same code. The only differences are the starting weights (random for
    # scratch, SimCLR for finetune, set above) and the learning rate (below).
    # nn.Sequential(enc, nn.Linear(512, 10)): the encoder followed by a new layer that turns
    # the 512 features into 10 class scores. Here the whole network is trained.
    model = nn.Sequential(enc, nn.Linear(512, 10)).to(dev)
    # The chosen training images, with light random edits (shift and flip).
    train_ds = labeled_set(a.data, "train", supervised_train_transform(normalize=norm), a.frac, a.seed)
    # The data loader for training.
    # Batch size = 128, or the number of images if there are fewer than 128.
    # Why the min(...): with 1% of the labels there are only 50 images. drop_last=True skips
    # an incomplete last batch, and with a batch size of 128 the only batch (50 images) would
    # be skipped, so the model would never be trained and would stay at chance level (about
    # 10% accuracy). Capping the batch size at the number of images fixes this.
    dl = DataLoader(train_ds, min(128, len(train_ds)), shuffle=True, num_workers=4, drop_last=True)
    # 100 epochs unless --epochs was given.
    epochs = a.epochs or 100
    # AdamW optimiser for ALL weights of the network.
    # Learning rate: --lr if given; otherwise 1e-3 (0.001) for scratch and 3e-4 (0.0003) for
    # finetune. The smaller rate for fine-tuning adjusts the pretrained weights gently
    # instead of overwriting what SimCLR learned.
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr or (1e-3 if a.mode == "scratch" else 3e-4), weight_decay=1e-4)
    # Cosine learning-rate schedule over all updates (epochs x batches per epoch).
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(dl))
    # Training loop.
    for ep in range(epochs):
        # Training mode.
        model.train()
        for x, y in dl:
            # Scores for the batch, compared with the true labels using cross-entropy.
            loss = nn.functional.cross_entropy(model(x.to(dev)), y.to(dev))
            # Clear old gradients, compute new ones, update the weights, advance the schedule.
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    # Accuracy on the 8,000 test images.
    result = acc(model, test_dl)

# ---------------- record the result ----------------
# Print the result, e.g. "linear tag=simclr_full frac=0.1 seed=0 test_acc=0.6522".
print(f"{a.mode} tag={a.tag} frac={a.frac} seed={a.seed} test_acc={result:.4f}")
# Make sure the results folder exists.
os.makedirs("results", exist_ok=True)
# Check whether results.csv is new, so the header row is written only once.
new_file = not os.path.exists("results/results.csv")
# "a" = append: add a line to the end of the file without deleting earlier results.
with open("results/results.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new_file:
        # Column names, written only when the file is created.
        w.writerow(["mode", "tag", "frac", "seed", "test_acc"])
    # One line for this run. f"{result:.4f}" writes the accuracy with 4 decimals (0.6522).
    w.writerow([a.mode, a.tag, a.frac, a.seed, f"{result:.4f}"])
