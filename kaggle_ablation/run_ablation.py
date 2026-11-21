"""Kaggle launcher: augmentation ablation for SimCLR (the project's core question).

For each setting below we (1) pretrain SimCLR for a SHORT schedule with only those
augmentations switched on, then (2) measure a linear probe on 100% of the labels.
Every setting uses the same schedule, the same unlabelled images and the same seed,
so differences in accuracy come only from the augmentations.

Results are copied to /kaggle/working after EVERY setting, so a run that is cut off
still leaves the finished results.
"""
import os, shutil, subprocess, sys

EPOCHS = 20      # short schedule, identical for every setting
SUBSET = 20000   # number of unlabelled images used, identical for every setting
BATCH = 256

REPO_URL = "https://github.com/ShubhamKang/SimCLR-Deep-Learning.git"
REPO_DIR = "/kaggle/temp/repo"   # scratch space (not saved)
DATA_DIR = "/kaggle/temp/data"   # STL-10 download (not saved, it is large)
OUT_DIR = "/kaggle/working"      # saved as the notebook output

# name -> augmentations that are ON. Most informative settings come first, so if time
# runs out we still have the key comparisons. "none" = no augmentation except resizing.
SETTINGS = [
    ("full",       ["crop", "flip", "color", "gray", "blur"]),
    ("no_crop",    ["flip", "color", "gray", "blur"]),
    ("no_color",   ["crop", "flip", "gray", "blur"]),
    ("only_crop",  ["crop"]),
    ("only_color", ["color"]),
    ("crop_color", ["crop", "color"]),
    ("no_gray",    ["crop", "flip", "color", "blur"]),
    ("no_blur",    ["crop", "flip", "color", "gray"]),
    ("no_flip",    ["crop", "color", "gray", "blur"]),
    ("none",       []),
]


def run(cmd, cwd=None):
    """Run a command, show its output live, and raise an error if it fails."""
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


# 1. Get the code from GitHub (branch main).
os.makedirs("/kaggle/temp", exist_ok=True)
run(["rm", "-rf", REPO_DIR])
run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
run(["nvidia-smi", "-L"])

# 2. Loop over the settings.
for name, augs in SETTINGS:
    tag = f"abl_{name}"
    # Pretrain. "--augs" with no names after it means every augmentation is off.
    # --out_dir REPO_DIR keeps checkpoints inside the repo copy so evaluate.py finds them.
    run([sys.executable, "train_simclr.py", "--name", tag, "--augs", *augs,
         "--epochs", str(EPOCHS), "--subset", str(SUBSET), "--batch_size", str(BATCH),
         "--data", DATA_DIR, "--out_dir", REPO_DIR, "--workers", "4"], cwd=REPO_DIR)
    # Linear probe on all 5000 labels; appends one row to results/results.csv.
    run([sys.executable, "evaluate.py", "--mode", "linear",
         "--ckpt", f"{REPO_DIR}/checkpoints/{tag}.pt", "--frac", "1.0",
         "--tag", tag, "--data", DATA_DIR], cwd=REPO_DIR)
    # Save progress: copy the results table and the pretraining loss logs to the output.
    os.makedirs(f"{OUT_DIR}/results/logs", exist_ok=True)
    shutil.copy(f"{REPO_DIR}/results/results.csv", f"{OUT_DIR}/results/results_ablation.csv")
    shutil.copy(f"{REPO_DIR}/results/logs/pretrain_{tag}.csv", f"{OUT_DIR}/results/logs/")
    # Delete the checkpoint (45 MB each) so the output stays small; only results matter here.
    os.remove(f"{REPO_DIR}/checkpoints/{tag}.pt")
