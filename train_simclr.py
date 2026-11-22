"""SimCLR pretraining on the STL-10 unlabelled images.

What this script does, step by step:
  1. Loads unlabelled images; each item is two random augmented views of one image.
  2. Passes both views through the encoder + projection head.
  3. Computes the NT-Xent contrastive loss (views of the same image should match).
  4. Updates the network, and saves the encoder after every epoch.

No labels are used anywhere in this file.

Examples:
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

# ---------------- command-line options ----------------
p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="run name; used for the checkpoint and log file names")
p.add_argument("--augs", nargs="*", default=ALL_AUGS, help="augmentations to switch ON, from: " + " ".join(ALL_AUGS))
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch_size", type=int, default=256, help="images per batch; each gives 2 views, so the network sees 2x this")
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--temp", type=float, default=0.5, help="loss temperature")
p.add_argument("--subset", type=int, default=None, help="use only N unlabelled images (for fast ablations)")
p.add_argument("--data", default="data", help="folder where STL-10 is stored / downloaded")
p.add_argument("--workers", type=int, default=4, help="CPU processes that prepare augmented images")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--resume", action="store_true", help="continue from the last saved epoch if a state file exists in out_dir")
p.add_argument("--out_dir", default=".", help="where checkpoints/ and results/logs/ are written (Kaggle: /kaggle/working)")
a = p.parse_args()

# ---------------- setup ----------------
torch.manual_seed(a.seed)  # reproducible weight initialisation and shuffling
dev = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset yields (view1, view2). drop_last=True keeps every batch the same size, which
# keeps the number of negatives per image constant.
ds = unlabeled_set(a.data, simclr_transform(a.augs), a.subset, a.seed)
dl = DataLoader(ds, a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers,
                pin_memory=True, persistent_workers=a.workers > 0)

model = SimCLR().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
# Cosine schedule: the learning rate decays smoothly to ~0 over all training steps.
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(dl))
# GradScaler is needed for mixed precision (fp16) on GPU; it is disabled on CPU.
scaler = torch.amp.GradScaler(enabled=dev == "cuda")

ckpt_dir = os.path.join(a.out_dir, "checkpoints")   # encoder weights are saved here
log_dir = os.path.join(a.out_dir, "results", "logs")  # per-epoch loss CSV is saved here
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"pretrain_{a.name}.csv")
# The state file holds EVERYTHING needed to continue training after a crash or disconnect:
# full model, optimiser, learning-rate schedule, grad scaler and the last finished epoch.
state_path = os.path.join(ckpt_dir, f"{a.name}_state.pt")
start_ep = 1  # first epoch to train; changes only when resuming
if a.resume and os.path.exists(state_path):
    # Load the saved state onto the same device we train on.
    st = torch.load(state_path, map_location=dev)
    model.load_state_dict(st["model"])    # encoder + projection head weights
    opt.load_state_dict(st["opt"])        # AdamW moment estimates, so updates continue smoothly
    sched.load_state_dict(st["sched"])    # position in the cosine learning-rate schedule
    scaler.load_state_dict(st["scaler"])  # mixed-precision loss-scale value
    start_ep = st["epoch"] + 1            # continue after the last finished epoch
    print(f"Resuming from epoch {start_ep}", flush=True)
    # The existing loss log is kept and new epochs are appended to it.
else:
    # Fresh run: create the loss log with its header row (overwrites an old log of the same name).
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "seconds"])  # header; one row is added per epoch

# ---------------- training loop ----------------
t0 = time.time()
for ep in range(start_ep, a.epochs + 1):
    model.train()
    tot = 0
    for x1, x2 in tqdm(dl, desc=f"ep {ep}/{a.epochs}", leave=False):
        x1, x2 = x1.to(dev, non_blocking=True), x2.to(dev, non_blocking=True)

        # Forward both views in ONE pass (concatenated) for speed. Mixed precision
        # (autocast) uses fp16 where safe, which is faster and uses less GPU memory.
        with torch.autocast(dev, enabled=dev == "cuda"):
            z = model(torch.cat([x1, x2]))

        # Split the outputs back into view 1 and view 2 and compute the contrastive loss.
        loss = nt_xent(z[: len(x1)], z[len(x1):], a.temp)

        # Standard update: clear old gradients, backpropagate (scaled for fp16),
        # step the optimiser, then step the learning-rate schedule.
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item()

    tot /= len(dl)  # average loss over the epoch
    print(f"epoch {ep} loss {tot:.4f} time {time.time()-t0:.0f}s", flush=True)
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([ep, tot, int(time.time() - t0)])

    # Save ONLY the encoder (the projection head is discarded after pretraining) and
    # overwrite every epoch, so a crashed session (e.g. Colab) loses little work.
    torch.save({"encoder": model.encoder.state_dict(), "args": vars(a), "epoch": ep}, os.path.join(ckpt_dir, f"{a.name}.pt"))
    # Full state for --resume (written after the encoder file so a crash never loses the encoder).
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": ep}, state_path)
