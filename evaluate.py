"""Evaluate on STL-10 test with few labels.
  --mode scratch   supervised ResNet-18, random init (baseline)
  --mode linear    frozen SimCLR encoder + linear classifier
  --mode finetune  SimCLR encoder, all layers trained
Results are appended to results/results.csv. Example:
python evaluate.py --mode linear --ckpt checkpoints/full.pt --frac 0.1 --tag full
"""
import argparse, csv, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.augmentations import test_transform, supervised_train_transform
from src.data import labeled_set
from src.model import make_encoder

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["scratch", "linear", "finetune"], required=True)
p.add_argument("--ckpt", default=None)
p.add_argument("--frac", type=float, default=1.0)
p.add_argument("--tag", default="")
p.add_argument("--epochs", type=int, default=None)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default="data")
a = p.parse_args()
torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

enc = make_encoder()
if a.mode != "scratch":
    enc.load_state_dict(torch.load(a.ckpt, map_location="cpu")["encoder"])
enc.to(dev)

def acc(model, dl):
    model.eval(); c = n = 0
    with torch.no_grad():
        for x, y in dl:
            c += (model(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return c / n

test_dl = DataLoader(labeled_set(a.data, "test", test_transform()), 512, num_workers=4)

if a.mode == "linear":
    @torch.no_grad()
    def feats(dl):
        enc.eval(); F, Y = [], []
        for x, y in dl:
            F.append(enc(x.to(dev)).cpu()); Y.append(y)
        return torch.cat(F), torch.cat(Y)
    ftr, ytr = feats(DataLoader(labeled_set(a.data, "train", test_transform(), a.frac, a.seed), 512, num_workers=4))
    fte, yte = feats(test_dl)
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fte = ((ftr - mu) / sd).to(dev), ((fte - mu) / sd).to(dev)
    ytr = ytr.to(dev)
    clf = nn.Linear(512, 10).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=a.lr or 1e-3, weight_decay=1e-4)
    for _ in range(a.epochs or 300):
        loss = nn.functional.cross_entropy(clf(ftr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    result = (clf(fte).argmax(1).cpu() == yte).float().mean().item()
else:
    model = nn.Sequential(enc, nn.Linear(512, 10)).to(dev)
    dl = DataLoader(labeled_set(a.data, "train", supervised_train_transform(), a.frac, a.seed), 128, shuffle=True, num_workers=4, drop_last=True)
    epochs = a.epochs or 100
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr or (1e-3 if a.mode == "scratch" else 3e-4), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(dl))
    for ep in range(epochs):
        model.train()
        for x, y in dl:
            loss = nn.functional.cross_entropy(model(x.to(dev)), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    result = acc(model, test_dl)

print(f"{a.mode} tag={a.tag} frac={a.frac} seed={a.seed} test_acc={result:.4f}")
os.makedirs("results", exist_ok=True)
new = not os.path.exists("results/results.csv")
with open("results/results.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["mode", "tag", "frac", "seed", "test_acc"])
    w.writerow([a.mode, a.tag, a.frac, a.seed, f"{result:.4f}"])
