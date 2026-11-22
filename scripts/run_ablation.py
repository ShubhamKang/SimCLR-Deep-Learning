"""Run a chosen part of the ablations on one machine (used so teammates can split the work).

Each setting = (1) SimCLR pretraining on a short schedule, (2) a linear probe on 100% of the labels.
The schedule is the SAME for every setting and every person (20 epochs, 20,000 unlabelled images, seed 0),
so numbers from different machines can be combined into one table.

Examples (run from the repo root):
  python scripts/run_ablation.py --who sandhya --aug full no_crop no_color only_crop
  python scripts/run_ablation.py --who nidhi --bs 64 128 512

Output: results/ablation_<who>.csv (rows with tags abl_* or bs_*). Send or commit this file.
"""
import argparse, csv, os, subprocess, sys

# Augmentation settings: name -> augmentations that are ON (see src/augmentations.py).
AUG_SETTINGS = {
    "full":       ["crop", "flip", "color", "gray", "blur"],
    "no_crop":    ["flip", "color", "gray", "blur"],
    "no_color":   ["crop", "flip", "gray", "blur"],
    "no_gray":    ["crop", "flip", "color", "blur"],
    "no_blur":    ["crop", "flip", "color", "gray"],
    "no_flip":    ["crop", "color", "gray", "blur"],
    "only_crop":  ["crop"],
    "only_color": ["color"],
    "crop_color": ["crop", "color"],
    "none":       [],
}
EPOCHS, SUBSET = 20, 20000  # fixed short schedule; do not change, or results are not comparable

p = argparse.ArgumentParser()
p.add_argument("--who", required=True, help="your name; used in the output file name")
p.add_argument("--aug", nargs="*", default=[], help="augmentation settings to run: " + " ".join(AUG_SETTINGS))
p.add_argument("--bs", nargs="*", type=int, default=[], help="batch sizes to run (all augmentations on)")
p.add_argument("--workers", type=int, default=2, help="data-loading processes (Colab has 2 CPU cores)")
a = p.parse_args()


def run(cmd):
    """Run a command, show its output live, stop if it fails."""
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def save_rows():
    """Copy the ablation rows from results/results.csv into results/ablation_<who>.csv.
    Baseline rows (supervised etc.) are left out so merging files later causes no duplicates."""
    with open("results/results.csv") as f:
        rows = list(csv.reader(f))
    keep = [rows[0]] + [r for r in rows[1:] if r[1].startswith(("abl_", "bs_"))]
    with open(f"results/ablation_{a.who}.csv", "w", newline="") as f:
        csv.writer(f).writerows(keep)


def one_run(tag, augs, batch):
    """Pretrain with the given augmentations and batch size, then linear-probe the encoder."""
    run([sys.executable, "train_simclr.py", "--name", tag, "--augs", *augs, "--epochs", str(EPOCHS),
         "--subset", str(SUBSET), "--batch_size", str(batch), "--workers", str(a.workers)])
    run([sys.executable, "evaluate.py", "--mode", "linear", "--ckpt", f"checkpoints/{tag}.pt",
         "--frac", "1.0", "--tag", tag])
    save_rows()  # save after every run so a disconnect loses at most one run


for name in a.aug:
    one_run(f"abl_{name}", AUG_SETTINGS[name], 256)
for bs in a.bs:
    one_run(f"bs_{bs}", AUG_SETTINGS["full"], bs)
print("Done. Send results/ablation_%s.csv to Shubham." % a.who)
