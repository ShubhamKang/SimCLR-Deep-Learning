"""Kaggle launcher: Sandhya's share of the augmentation ablation (run on Kaggle).

Runs the six augmentation settings that are NOT assigned to Sandhya, using the same
scripts/run_ablation.py as everyone else (same 20-epoch / 20k-image schedule), so the
results can be merged into one table. Kaggle runs this unattended once a GPU is free.
"""
# os works with folders, shutil copies files, subprocess runs other programs
# (git, python), sys gives the path of the running Python.
import os, shutil, subprocess, sys

# Address of our code on GitHub. Kaggle downloads a fresh copy of it at the start of every run.
REPO_URL = "https://github.com/ShubhamKang/SimCLR-Deep-Learning.git"
REPO_DIR = "/kaggle/temp/repo"   # scratch space: code + STL-10 download (not saved)
OUT_DIR = "/kaggle/working"      # saved as the notebook output
SETTINGS = ["full", "no_crop", "no_color", "only_crop"]  # Sandhya's share, run on Kaggle


def run(cmd, cwd=None):
    """Run a command, show its output live, and stop if it fails."""
    # Print the command first, so the Kaggle log shows what is running.
    print("$", " ".join(cmd), flush=True)
    # Run it inside the folder cwd. check=True stops the whole script if the command fails.
    subprocess.run(cmd, cwd=cwd, check=True)


# 1. Fresh copy of the code from GitHub, and print the GPU model for the report.
os.makedirs("/kaggle/temp", exist_ok=True)
# Delete any old copy of the code, so the newest version from GitHub is used.
run(["rm", "-rf", REPO_DIR])
# Download the code from GitHub. --depth 1 fetches only the latest version (faster).
run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
# Print which GPU Kaggle gave us (for example Tesla T4).
run(["nvidia-smi", "-L"])

# 2. One setting at a time, copying the results to the saved output after each one,
#    so a run that is stopped early still keeps every finished setting.
os.makedirs(f"{OUT_DIR}/results/logs", exist_ok=True)
for name in SETTINGS:
    # --workers 4: Kaggle machines have 4 CPU cores for preparing augmented images.
    run([sys.executable, "scripts/run_ablation.py", "--who", "sandhya", "--aug", name, "--workers", "4"], cwd=REPO_DIR)
    # Copy the results so far, and this run's loss log, into the saved output folder.
    shutil.copy(f"{REPO_DIR}/results/ablation_sandhya.csv", f"{OUT_DIR}/results/")
    shutil.copy(f"{REPO_DIR}/results/logs/pretrain_abl_{name}.csv", f"{OUT_DIR}/results/logs/")
    os.remove(f"{REPO_DIR}/checkpoints/abl_{name}.pt")  # 45 MB each, not needed after the probe

# 3. Batch-size ablation (Nidhi's share) with all augmentations on. The batch-256 point is "full" above.
for bs in [64, 128, 512]:
    run([sys.executable, "scripts/run_ablation.py", "--who", "nidhi", "--bs", str(bs), "--workers", "4"], cwd=REPO_DIR)
    shutil.copy(f"{REPO_DIR}/results/ablation_nidhi.csv", f"{OUT_DIR}/results/")
    shutil.copy(f"{REPO_DIR}/results/logs/pretrain_bs_{bs}.csv", f"{OUT_DIR}/results/logs/")
    os.remove(f"{REPO_DIR}/checkpoints/bs_{bs}.pt")
