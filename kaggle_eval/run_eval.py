"""Kaggle launcher: evaluate the finished SimCLR checkpoint (linear probe + fine-tune).

It reads full.pt from the OUTPUT of the simclr-pretrain-full kernel, which Kaggle mounts
read-only under /kaggle/input/ (see "kernel_sources" in kernel-metadata.json).
Results are copied to /kaggle/working after every evaluation, so partial results survive.
"""
# glob finds files by pattern, os works with folders, shutil copies files,
# subprocess runs other programs (git, python), sys gives the path of the running Python.
import glob, os, shutil, subprocess, sys

# Address of our code on GitHub. Kaggle downloads a fresh copy of it at the start of every run.
REPO_URL = "https://github.com/ShubhamKang/SimCLR-Deep-Learning.git"
# Scratch folder for the code (not saved after the run).
REPO_DIR = "/kaggle/temp/repo"
# Everything written here is saved by Kaggle as the notebook's output.
OUT_DIR = "/kaggle/working"


def run(cmd, cwd=None):
    """Run a command, show its output live, and stop if it fails."""
    # Print the command first, so the Kaggle log shows what is running.
    print("$", " ".join(cmd), flush=True)
    # Run it inside the folder cwd. check=True stops the whole script if the command fails.
    subprocess.run(cmd, cwd=cwd, check=True)


# Create the scratch folder (no error if it already exists).
os.makedirs("/kaggle/temp", exist_ok=True)
# Delete any old copy of the code, so the newest version from GitHub is used.
run(["rm", "-rf", REPO_DIR])
# Download the code from GitHub. --depth 1 fetches only the latest version (faster).
run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])

# Find full.pt inside the mounted output of the pretraining kernel and copy it next to the code.
ckpt = glob.glob("/kaggle/input/**/checkpoints/full.pt", recursive=True)[0]
# glob(..., recursive=True) searches all sub-folders; [0] takes the first match.
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
        # After every evaluation, copy the results so far to the saved output.
        shutil.copy(f"{REPO_DIR}/results/results.csv", f"{OUT_DIR}/results_simclr_full.csv")
