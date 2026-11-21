"""Kaggle launcher: evaluate the finished SimCLR checkpoint (linear probe + fine-tune).

It reads full.pt from the OUTPUT of the simclr-pretrain-full kernel, which Kaggle mounts
read-only under /kaggle/input/ (see "kernel_sources" in kernel-metadata.json).
Results are copied to /kaggle/working after every evaluation, so partial results survive.
"""
import glob, os, shutil, subprocess, sys

REPO_URL = "https://github.com/ShubhamKang/SimCLR-Deep-Learning.git"
REPO_DIR = "/kaggle/temp/repo"
OUT_DIR = "/kaggle/working"


def run(cmd, cwd=None):
    """Run a command, show its output live, and stop if it fails."""
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


os.makedirs("/kaggle/temp", exist_ok=True)
run(["rm", "-rf", REPO_DIR])
run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])

# Find full.pt inside the mounted output of the pretraining kernel and copy it next to the code.
ckpt = glob.glob("/kaggle/input/**/checkpoints/full.pt", recursive=True)[0]
os.makedirs(f"{REPO_DIR}/checkpoints", exist_ok=True)
shutil.copy(ckpt, f"{REPO_DIR}/checkpoints/full.pt")
print("Using checkpoint", ckpt, flush=True)

# Start a fresh results file so only this run's rows are in the output.
os.makedirs(f"{REPO_DIR}/results", exist_ok=True)
if os.path.exists(f"{REPO_DIR}/results/results.csv"):
    os.remove(f"{REPO_DIR}/results/results.csv")

# Linear probe first (fast), then fine-tuning, at every label fraction used by the baseline.
for mode in ["linear", "finetune"]:
    for frac in ["1.0", "0.2", "0.1", "0.01"]:
        run([sys.executable, "evaluate.py", "--mode", mode, "--ckpt", "checkpoints/full.pt",
             "--frac", frac, "--tag", "simclr_full"], cwd=REPO_DIR)
        shutil.copy(f"{REPO_DIR}/results/results.csv", f"{OUT_DIR}/results_simclr_full.csv")
