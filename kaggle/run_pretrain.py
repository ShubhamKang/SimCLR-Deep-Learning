"""Kaggle launcher: runs SimCLR pretraining on a Kaggle GPU.

This file is pushed to Kaggle with `kaggle kernels push`. Kaggle runs it top to
bottom on a GPU machine, even if your browser is closed. Everything saved in
/kaggle/working becomes the notebook "Output", which you can download afterwards.
"""
import os, subprocess, sys

# ---- settings you can change ----
EPOCHS = int(os.environ.get("EPOCHS", 50))   # how many passes over the unlabelled images
BATCH = int(os.environ.get("BATCH", 256))    # images per batch (each gives 2 views)
NAME = "full"                                # run name -> checkpoints/full.pt

# Folders. /kaggle/temp is scratch space that is NOT saved after the run, so the big
# STL-10 download (about 2.6 GB) and the git clone stay out of the saved output.
# /kaggle/working IS saved, so the checkpoint and the loss log go there.
REPO_URL = "https://github.com/ShubhamKang/SimCLR-Deep-Learning.git"
REPO_DIR = "/kaggle/temp/repo"
DATA_DIR = "/kaggle/temp/data"
OUT_DIR = "/kaggle/working"


def run(cmd, cwd=None):
    """Run a shell command, print its output live, and stop everything if it fails."""
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)  # check=True raises an error on failure


# 1. Download our code from GitHub (branch main). Remove an old copy first if present.
os.makedirs("/kaggle/temp", exist_ok=True)
if os.path.exists(REPO_DIR):
    run(["rm", "-rf", REPO_DIR])
run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])

# 2. Show which GPU we got (useful to note in the report).
run(["nvidia-smi", "-L"])

# 3. Start pretraining. Arguments:
#    --data    where STL-10 is downloaded to
#    --out_dir where checkpoints/full.pt and results/logs/pretrain_full.csv are written
#    --workers number of CPU processes that prepare augmented images (Kaggle has 4 cores)
# The script overwrites full.pt after EVERY epoch, so if Kaggle stops the run early,
# the latest epoch is still in the output.
run([sys.executable, "train_simclr.py", "--name", NAME, "--epochs", str(EPOCHS),
     "--batch_size", str(BATCH), "--data", DATA_DIR, "--out_dir", OUT_DIR, "--workers", "4"],
    cwd=REPO_DIR)

# 4. List the saved files so they appear in the run log.
run(["ls", "-lh", f"{OUT_DIR}/checkpoints", f"{OUT_DIR}/results/logs"])
