"""SimCLR pretraining on the unlabelled STL-10 images.

This script teaches the ResNet-18 encoder what images look like, WITHOUT using any labels:
  1. Load the unlabelled images. Each item is two randomly edited views of one image.
  2. Pass both views through the encoder and the projection head.
  3. Calculate the NT-Xent loss, which is small when the two views of the same image get
     similar outputs and views of different images get different outputs.
  4. Change the weights a little to make the loss smaller ("one update"), and repeat.
  5. After every epoch (one pass over all images), save the encoder to a file.

Words used below:
  epoch      : one full pass over all the training images.
  batch      : a group of images processed together; one weight update per batch.
  optimiser  : the rule that decides how to change the weights after each batch.
  learning rate (lr) : how big each weight change is.

How to run it (examples):
  Main model used in the report (50 epochs on all 100,000 images, batch 256):
    python train_simclr.py --name full --epochs 50 --batch_size 256
  An augmentation-ablation run (blur switched off, 20 epochs on 20,000 images):
    python train_simclr.py --name abl_no_blur --augs crop flip color gray --epochs 20 --subset 20000
"""
# argparse reads options typed on the command line (like --epochs 50).
# csv writes comma-separated text files (our loss log). os handles folders and file paths.
# time measures how long training takes.
import argparse, csv, os, time
# torch is PyTorch, the deep learning library.
import torch
# DataLoader takes a dataset and serves it in shuffled batches.
from torch.utils.data import DataLoader
# tqdm draws a progress bar in the terminal.
from tqdm import tqdm
# Our own code from the src/ folder.
from src.augmentations import simclr_transform, ALL_AUGS
from src.data import unlabeled_set
from src.model import SimCLR
from src.losses import nt_xent

# ---------------- command-line options ----------------
# Each add_argument line defines one option: its name, its type, its default value (used if
# the option is not typed) and a help text. Example: --epochs 50 sets a.epochs to 50.
p = argparse.ArgumentParser()
# Name of the run, used in the file names of the checkpoint and the loss log.
p.add_argument("--name", required=True, help="run name; used for the checkpoint and log file names")
# Which augmentations to switch on. nargs="*" means "a list of any length", e.g. --augs crop color.
p.add_argument("--augs", nargs="*", default=ALL_AUGS, help="augmentations to switch ON, from: " + " ".join(ALL_AUGS))
# Number of epochs. The default is 100, but the report's main run used --epochs 50.
p.add_argument("--epochs", type=int, default=100)
# Images per batch. Each image gives two views, so the network sees twice this many per step.
p.add_argument("--batch_size", type=int, default=256, help="images per batch; each gives 2 views, so the network sees 2x this")
# Learning rate: 3e-4 means 0.0003.
p.add_argument("--lr", type=float, default=3e-4)
# Temperature of the NT-Xent loss (see src/losses.py).
p.add_argument("--temp", type=float, default=0.5, help="loss temperature")
# Use only this many unlabelled images (the ablations use 20,000). None means all 100,000.
p.add_argument("--subset", type=int, default=None, help="use only N unlabelled images (for fast ablations)")
# Folder where STL-10 is stored or downloaded to.
p.add_argument("--data", default="data", help="folder where STL-10 is stored / downloaded")
# Number of CPU processes that prepare the edited images in parallel while the GPU trains.
p.add_argument("--workers", type=int, default=4, help="CPU processes that prepare augmented images")
# Seed for the random number generators, so runs can be repeated.
p.add_argument("--seed", type=int, default=0)
# If given, continue a run that was interrupted instead of starting from scratch.
p.add_argument("--resume", action="store_true", help="continue from the last saved epoch if a state file exists in out_dir")
# Folder where the checkpoints and logs are written.
p.add_argument("--out_dir", default=".", help="where checkpoints/ and results/logs/ are written (Kaggle: /kaggle/working)")
# Read the options that were actually typed. They are then available as a.name, a.epochs, ...
a = p.parse_args()

# ---------------- setup ----------------
# Fix the seed of PyTorch's random number generator, so the random starting weights and the
# shuffling order are the same each time the script runs with the same seed.
torch.manual_seed(a.seed)
# Use the GPU ("cuda") if one is available, otherwise the CPU. A GPU is many times faster.
dev = "cuda" if torch.cuda.is_available() else "cpu"

# Build the dataset: unlabelled images, each returned as two views made with the chosen edits.
ds = unlabeled_set(a.data, simclr_transform(a.augs), a.subset, a.seed)
# The DataLoader serves the dataset in batches:
#   a.batch_size       : images per batch
#   shuffle=True       : a new random order every epoch
#   drop_last=True     : if the last batch is smaller than batch_size, skip it, so every batch
#                        has the same size (and so the same number of negatives in the loss)
#   num_workers        : CPU processes that load and edit images in parallel
#   pin_memory=True    : speeds up copying batches from the CPU to the GPU
#   persistent_workers : keep the worker processes alive between epochs (faster)
dl = DataLoader(ds, a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers,
                pin_memory=True, persistent_workers=a.workers > 0)

# Create the network (ResNet-18 encoder + projection head) and move it to the GPU or CPU.
model = SimCLR().to(dev)
# The optimiser AdamW changes the weights after every batch. lr is the learning rate.
# AdamW (Adam with decoupled Weight decay): Adam gives every weight its own step size, based
# on running averages of its recent gradients (gradient = the direction that reduces the
# loss); the "W" means weight decay is applied separately from that gradient step.
# weight_decay=1e-6 (0.000001) gently pulls all weights towards zero, which helps against
# overfitting. model.parameters() is the list of all weights it is allowed to change.
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
# Cosine learning-rate schedule: the learning rate starts at a.lr and falls smoothly to almost
# zero by the end of training, following the shape of a cosine curve. Big steps early, small
# careful steps at the end. The second number is the total number of updates:
# epochs times batches per epoch (len(dl)).
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(dl))
# Mixed precision means doing some calculations with 16-bit numbers instead of 32-bit, which
# is faster and uses less GPU memory. Very small numbers can become 0 in 16-bit, so the
# GradScaler multiplies the loss by a large factor before computing gradients and undoes it
# afterwards. It is only switched on when a GPU is used.
scaler = torch.amp.GradScaler(enabled=dev == "cuda")

# Folder for saved weights ("checkpoints"), e.g. ./checkpoints
ckpt_dir = os.path.join(a.out_dir, "checkpoints")
# Folder for the loss log, e.g. ./results/logs
log_dir = os.path.join(a.out_dir, "results", "logs")
# Create both folders. exist_ok=True means "no error if they already exist".
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
# Path of the loss log for this run, e.g. results/logs/pretrain_full.csv
log_path = os.path.join(log_dir, f"pretrain_{a.name}.csv")
# Path of the "state" file. It holds everything needed to continue training after an
# interruption (for example a lost internet connection on Colab): the full network, the
# optimiser, the learning-rate schedule, the GradScaler and the last finished epoch.
state_path = os.path.join(ckpt_dir, f"{a.name}_state.pt")
# The epoch to start from. 1 for a new run; changed below when resuming.
start_ep = 1
# Resume only if --resume was given AND a state file from an earlier run exists.
if a.resume and os.path.exists(state_path):
    # Read the saved state from disk onto the device we train on.
    st = torch.load(state_path, map_location=dev)
    # Put every saved part back:
    model.load_state_dict(st["model"])    # the weights of the encoder and the projection head
    opt.load_state_dict(st["opt"])        # the optimiser's internal running averages
    sched.load_state_dict(st["sched"])    # where we are on the learning-rate curve
    scaler.load_state_dict(st["scaler"])  # the GradScaler's current scaling factor
    # Continue with the epoch after the last one that finished.
    start_ep = st["epoch"] + 1
    # flush=True prints immediately instead of waiting in a buffer.
    print(f"Resuming from epoch {start_ep}", flush=True)
    # The existing loss log is kept; new epochs are added to the end of it.
else:
    # New run: create the loss log and write its header row (the column names).
    # "w" = write mode, which replaces any old file with the same name.
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "seconds"])

# ---------------- training loop ----------------
# Remember the start time, to report how long training has taken.
t0 = time.time()
# Loop over the epochs. range(start_ep, a.epochs + 1) gives start_ep, ..., a.epochs.
for ep in range(start_ep, a.epochs + 1):
    # Put the network in training mode (some layers, such as batch normalisation, behave
    # differently during training and evaluation).
    model.train()
    # Running total of the loss in this epoch, to compute the average at the end.
    tot = 0
    # Loop over the batches. Each batch gives x1 (first views) and x2 (second views).
    # tqdm(...) only wraps the loop to show a progress bar.
    for x1, x2 in tqdm(dl, desc=f"ep {ep}/{a.epochs}", leave=False):
        # Move the images to the GPU. non_blocking=True lets the copy overlap with other work.
        x1, x2 = x1.to(dev, non_blocking=True), x2.to(dev, non_blocking=True)

        # torch.autocast switches on mixed precision (16-bit where it is safe) on the GPU.
        with torch.autocast(dev, enabled=dev == "cuda"):
            # Put both sets of views together into one batch of 2 x batch_size images and
            # pass it through the network once. This is faster than two separate passes.
            # z holds the 128 output numbers for every view.
            z = model(torch.cat([x1, x2]))

        # Split z back into the outputs of the first views (first half) and the second views
        # (second half), and calculate the contrastive loss.
        # z[: len(x1)] = rows 0 to batch_size-1, z[len(x1):] = the remaining rows.
        loss = nt_xent(z[: len(x1)], z[len(x1):], a.temp)

        # The standard update step:
        # 1. Clear the gradients left over from the previous batch.
        opt.zero_grad(set_to_none=True)
        # 2. Backpropagation: compute how each weight should change to reduce the loss
        #    (the "gradients"). The loss is scaled first for mixed precision.
        scaler.scale(loss).backward()
        # 3. Let the optimiser change the weights using those gradients (the scaler undoes
        #    its scaling first and skips the step if something overflowed).
        scaler.step(opt)
        # 4. Let the scaler adjust its scaling factor for the next batch.
        scaler.update()
        # 5. Move one step along the cosine learning-rate curve.
        sched.step()
        # Add this batch's loss to the running total. .item() turns a one-number tensor into
        # a normal Python number.
        tot += loss.item()

    # Average loss over all batches of this epoch.
    tot /= len(dl)
    # Print the epoch number, its average loss and the time since the start, in seconds.
    print(f"epoch {ep} loss {tot:.4f} time {time.time()-t0:.0f}s", flush=True)
    # Add one row to the loss log. "a" = append mode, which adds to the end of the file.
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([ep, tot, int(time.time() - t0)])

    # Save the encoder's weights after every epoch, replacing the previous file, so an
    # interruption loses at most one epoch of work. Only the encoder is saved here, because
    # the projection head is thrown away after pretraining. This file (for example
    # checkpoints/full.pt) is what evaluate.py loads later. "args" stores the options used.
    torch.save({"encoder": model.encoder.state_dict(), "args": vars(a), "epoch": ep}, os.path.join(ckpt_dir, f"{a.name}.pt"))
    # Also save the full state for --resume. It is written after the encoder file, so even if
    # the session dies during this step, the encoder file is already safe.
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": ep}, state_path)
