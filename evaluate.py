"""Measure test accuracy on STL-10 when only a few labels are available.

Three ways to build the classifier (--mode):
  scratch   : supervised ResNet-18 with random weights, trained on the labels only.
              This is the BASELINE that shows what you get without pretraining.
  linear    : "linear probe". Freeze the SimCLR encoder, and train only one linear
              layer on its features. Measures how good the learned features are.
  finetune  : start from the SimCLR encoder and train ALL layers on the labels.

--frac sets how much of the 5,000 training labels is used (1.0, 0.2, 0.1, 0.01 ...).
Accuracy is always measured on the full 8,000-image test set.
Every run appends one row to results/results.csv.

Example:
  python evaluate.py --mode linear --ckpt checkpoints/full.pt --frac 0.1 --tag simclr_full

Results in the report (Table 1, test accuracy %):
  labels   scratch  linear  finetune
  100%      75.5     72.6     81.3
  20%       53.4     68.6     71.2
  10%       44.3     65.2     65.5
  1%        26.8     53.1     43.6

QUESTIONS THIS FILE ANSWERS
Q: How is the comparison kept fair?
   Same ResNet-18, same labelled images (seed 0, class-balanced), same 8,000 test images.
   Scratch and finetune share the same training recipe (100 epochs, AdamW, batch 128,
   cosine decay, light augmentation); only the starting weights and the learning rate differ.
Q: Why does the linear probe beat fine-tuning at 1% labels (53.1 vs 43.6)?
   With 50 images, fine-tuning all ~11 million weights for 100 epochs overfits and damages the
   pretrained features. The probe trains only 512*10 + 10 = 5,130 numbers, so it cannot
   overfit as badly.
Q: Why is the linear probe below the baseline at 100% (72.6 vs 75.5)?
   With 5,000 labels the baseline has enough data to learn task-specific features end to
   end, while the probe uses frozen features from only 50 epochs of pretraining.
   Fine-tuning closes the gap (81.3).
Q: Why use the linear probe to evaluate the ablations?
   It measures the learned features directly (encoder frozen) and is cheap, so it can be
   repeated for every ablation run; fine-tuning could hide weak features.
Q: What was the 1% bug?
   See the comment above the DataLoader in the fine-tune/scratch branch below.
"""
import argparse, csv, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.augmentations import test_transform, supervised_train_transform
from src.data import labeled_set
from src.model import make_encoder

# ---------------- command-line options ----------------
p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["scratch", "linear", "finetune"], required=True)
p.add_argument("--ckpt", default=None, help="SimCLR checkpoint (needed for linear and finetune)")
p.add_argument("--frac", type=float, default=1.0, help="fraction of training labels to use")
p.add_argument("--tag", default="", help="label for this run in results.csv, used by make_plots.py")
p.add_argument("--epochs", type=int, default=None, help="override default epochs")  # defaults: 300 probe steps, 100 epochs otherwise
p.add_argument("--lr", type=float, default=None, help="override default learning rate")
p.add_argument("--seed", type=int, default=0, help="changes which labelled images are picked and the init")
p.add_argument("--data", default="data")
p.add_argument("--no_normalize", action="store_true",
               help="skip mean/std normalisation (for the public sthalles checkpoint, trained without it)")
a = p.parse_args()
torch.manual_seed(a.seed)
norm = not a.no_normalize  # True for our own checkpoints, False for the public one
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- build the encoder ----------------
enc = make_encoder()
if a.mode != "scratch":
    # Load the pretrained SimCLR weights. Scratch mode keeps the random initialisation.
    enc.load_state_dict(torch.load(a.ckpt, map_location="cpu")["encoder"])
enc.to(dev)


def acc(model, dl):
    """Fraction of images in dl that the model classifies correctly."""
    model.eval()
    correct = n = 0
    with torch.no_grad():  # no gradients needed when only measuring accuracy
        for x, y in dl:
            correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
            n += len(y)
    return correct / n


# The test set is the same for every experiment (all 8,000 images, no augmentation),
# so numbers are directly comparable.
test_dl = DataLoader(labeled_set(a.data, "test", test_transform(norm)), 512, num_workers=4)

if a.mode == "linear":
    # ---------- linear probe ----------
    @torch.no_grad()
    def feats(dl):
        """Run the FROZEN encoder once and collect features and labels."""
        enc.eval()
        F, Y = [], []
        for x, y in dl:
            F.append(enc(x.to(dev)).cpu())
            Y.append(y)
        return torch.cat(F), torch.cat(Y)

    # Features are computed once with no augmentation (the encoder is frozen anyway),
    # which makes the probe very fast to train.
    ftr, ytr = feats(DataLoader(labeled_set(a.data, "train", test_transform(norm), a.frac, a.seed), 512, num_workers=4))
    fte, yte = feats(test_dl)

    # Standardise each of the 512 features (subtract mean, divide by std) so they are on the
    # same scale, which makes the linear layer train quickly and stably. The mean and std
    # come from the TRAINING features only: using test statistics would leak information
    # from the test set into training.
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fte = ((ftr - mu) / sd).to(dev), ((fte - mu) / sd).to(dev)
    ytr = ytr.to(dev)

    # One linear layer 512 -> 10 classes, trained with full-batch Adam: every step uses ALL
    # training features at once (at most 5,000 x 512 numbers, which fits in memory), for
    # 300 steps, learning rate 1e-3, weight decay 1e-4. The encoder is never updated here.
    clf = nn.Linear(512, 10).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=a.lr or 1e-3, weight_decay=1e-4)
    for _ in range(a.epochs or 300):
        loss = nn.functional.cross_entropy(clf(ftr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Test accuracy = share of test images whose highest-scoring class is the true class.
    result = (clf(fte).argmax(1).cpu() == yte).float().mean().item()

else:
    # ---------- scratch baseline and fine-tuning (same training code) ----------
    # Encoder + a new linear classifier on top; the whole network is trained.
    model = nn.Sequential(enc, nn.Linear(512, 10)).to(dev)
    train_ds = labeled_set(a.data, "train", supervised_train_transform(normalize=norm), a.frac, a.seed)
    # THE 1% BUG (report Section 4.6): with --frac 0.01 there are only 50 images, fewer than
    # the batch size of 128. drop_last=True throws away the last incomplete batch, and here
    # that was the ONLY batch, so the model got zero updates and stayed at chance level
    # (about 10%). Fix: cap the batch size at the dataset size, so every fraction trains
    # for the full 100 epochs. The reported 1% baseline (26.8%) uses this fix.
    dl = DataLoader(train_ds, min(128, len(train_ds)), shuffle=True, num_workers=4, drop_last=True)
    epochs = a.epochs or 100  # 100 epochs over the labelled subset for both scratch and finetune
    # Fine-tuning uses a smaller learning rate than scratch (1e-3) so that it adjusts
    # the pretrained features gently instead of destroying them.
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr or (1e-3 if a.mode == "scratch" else 3e-4), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(dl))
    for ep in range(epochs):
        model.train()
        for x, y in dl:
            loss = nn.functional.cross_entropy(model(x.to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    result = acc(model, test_dl)

# ---------------- record the result ----------------
# One row per run in results/results.csv: mode, tag, label fraction, seed, test accuracy.
# make_plots.py reads these files to build the report tables and figures.
print(f"{a.mode} tag={a.tag} frac={a.frac} seed={a.seed} test_acc={result:.4f}")
os.makedirs("results", exist_ok=True)
new_file = not os.path.exists("results/results.csv")
with open("results/results.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(["mode", "tag", "frac", "seed", "test_acc"])  # header on first write
    w.writerow([a.mode, a.tag, a.frac, a.seed, f"{result:.4f}"])
