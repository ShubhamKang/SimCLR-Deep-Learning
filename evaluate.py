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
  python evaluate.py --mode linear --ckpt checkpoints/full.pt --frac 0.1 --tag full
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
p.add_argument("--epochs", type=int, default=None, help="override default epochs")
p.add_argument("--lr", type=float, default=None, help="override default learning rate")
p.add_argument("--seed", type=int, default=0, help="changes which labelled images are picked and the init")
p.add_argument("--data", default="data")
a = p.parse_args()
torch.manual_seed(a.seed)
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


# The test set is the same for every experiment, so numbers are directly comparable.
test_dl = DataLoader(labeled_set(a.data, "test", test_transform()), 512, num_workers=4)

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
    ftr, ytr = feats(DataLoader(labeled_set(a.data, "train", test_transform(), a.frac, a.seed), 512, num_workers=4))
    fte, yte = feats(test_dl)

    # Standardise each feature using TRAIN statistics only (no test leakage).
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fte = ((ftr - mu) / sd).to(dev), ((fte - mu) / sd).to(dev)
    ytr = ytr.to(dev)

    # One linear layer 512 -> 10 classes, trained with full-batch Adam.
    clf = nn.Linear(512, 10).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=a.lr or 1e-3, weight_decay=1e-4)
    for _ in range(a.epochs or 300):
        loss = nn.functional.cross_entropy(clf(ftr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    result = (clf(fte).argmax(1).cpu() == yte).float().mean().item()

else:
    # ---------- scratch baseline and fine-tuning (same training code) ----------
    # Encoder + a new linear classifier on top; the whole network is trained.
    model = nn.Sequential(enc, nn.Linear(512, 10)).to(dev)
    dl = DataLoader(labeled_set(a.data, "train", supervised_train_transform(), a.frac, a.seed),
                    128, shuffle=True, num_workers=4, drop_last=True)
    epochs = a.epochs or 100
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
print(f"{a.mode} tag={a.tag} frac={a.frac} seed={a.seed} test_acc={result:.4f}")
os.makedirs("results", exist_ok=True)
new_file = not os.path.exists("results/results.csv")
with open("results/results.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(["mode", "tag", "frac", "seed", "test_acc"])  # header on first write
    w.writerow([a.mode, a.tag, a.frac, a.seed, f"{result:.4f}"])
