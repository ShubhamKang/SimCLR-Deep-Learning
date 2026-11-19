"""SimCLR pretraining on STL-10 unlabeled. Example:
python train_simclr.py --name full --epochs 100 --batch_size 256
python train_simclr.py --name no_blur --augs crop flip color gray --epochs 20 --subset 20000
"""
import argparse, csv, os, time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.augmentations import simclr_transform, ALL_AUGS
from src.data import unlabeled_set
from src.model import SimCLR
from src.losses import nt_xent

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--augs", nargs="*", default=ALL_AUGS, help="subset of: " + " ".join(ALL_AUGS))
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--temp", type=float, default=0.5)
p.add_argument("--subset", type=int, default=None, help="use N unlabeled images (ablations)")
p.add_argument("--data", default="data")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ds = unlabeled_set(a.data, simclr_transform(a.augs), a.subset, a.seed)
dl = DataLoader(ds, a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
model = SimCLR().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(dl))
scaler = torch.amp.GradScaler(enabled=dev == "cuda")

os.makedirs("checkpoints", exist_ok=True); os.makedirs("results/logs", exist_ok=True)
log_path = f"results/logs/pretrain_{a.name}.csv"
with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "loss", "seconds"])
t0 = time.time()
for ep in range(1, a.epochs + 1):
    model.train(); tot = 0
    for x1, x2 in tqdm(dl, desc=f"ep {ep}/{a.epochs}", leave=False):
        x1, x2 = x1.to(dev, non_blocking=True), x2.to(dev, non_blocking=True)
        with torch.autocast(dev, enabled=dev == "cuda"):
            z = model(torch.cat([x1, x2]))
        loss = nt_xent(z[: len(x1)], z[len(x1):], a.temp)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item()
    tot /= len(dl)
    print(f"epoch {ep} loss {tot:.4f} time {time.time()-t0:.0f}s", flush=True)
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([ep, tot, int(time.time() - t0)])
    torch.save({"encoder": model.encoder.state_dict(), "args": vars(a), "epoch": ep}, f"checkpoints/{a.name}.pt")
