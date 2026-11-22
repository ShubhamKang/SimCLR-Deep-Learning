"""Run a chosen part of the ablation experiments on one machine.

An "ablation" changes one thing at a time to see how much it matters. Each run here does:
  1. SimCLR pretraining on a short schedule (train_simclr.py), with the chosen
     augmentations and batch size;
  2. a linear probe on 100% of the labels (evaluate.py) to measure how good the result is.
The schedule is the same for every run and every person (20 epochs, 20,000 unlabelled
images, seed 0), so results from different machines can be put in one table.

This script lets the team split the work, for example (run from the project folder):
  python scripts/run_ablation.py --who sandhya --aug full no_crop no_color only_crop
  python scripts/run_ablation.py --who nidhi --bs 64 128 512

Output: results/ablation_<who>.csv, with one row per finished run.
"""
# argparse reads command-line options; csv reads and writes CSV files; os handles paths;
# subprocess runs other programs (here our two Python scripts); sys gives the path of the
# Python program currently running (sys.executable).
import argparse, csv, os, subprocess, sys

# Every augmentation setting: its name and the list of augmentations switched ON.
# A Python dictionary maps each name (key) to its list (value).
AUG_SETTINGS = {
    "full":       ["crop", "flip", "color", "gray", "blur"],  # all five, the reference
    "no_crop":    ["flip", "color", "gray", "blur"],          # leave-one-out: crop removed
    "no_color":   ["crop", "flip", "gray", "blur"],           # colour jitter removed
    "no_gray":    ["crop", "flip", "color", "blur"],          # grayscale removed
    "no_blur":    ["crop", "flip", "color", "gray"],          # blur removed
    "no_flip":    ["crop", "color", "gray", "blur"],          # flip removed
    "only_crop":  ["crop"],                                   # a single augmentation
    "only_color": ["color"],                                  # a single augmentation
    "crop_color": ["crop", "color"],                          # the pair highlighted by the SimCLR paper
    "none":       [],                                         # no augmentation at all
}
# The short schedule used for every ablation run. Changing it would make the results
# incomparable with the others.
EPOCHS, SUBSET = 20, 20000

# ---------------- command-line options ----------------
p = argparse.ArgumentParser()
# Your name, used in the output file name (results/ablation_<who>.csv).
p.add_argument("--who", required=True, help="your name; used in the output file name")
# Which augmentation settings to run (names from AUG_SETTINGS); can be several.
p.add_argument("--aug", nargs="*", default=[], help="augmentation settings to run: " + " ".join(AUG_SETTINGS))
# Which batch sizes to run (with all augmentations on); can be several.
p.add_argument("--bs", nargs="*", type=int, default=[], help="batch sizes to run (all augmentations on)")
# CPU processes for loading images. Colab gives 2 CPU cores, so 2 by default.
p.add_argument("--workers", type=int, default=2, help="data-loading processes (Colab has 2 CPU cores)")
a = p.parse_args()


def run(cmd):
    """Run a command (a list of words), show its output live, and stop if it fails."""
    # Print the command first, so the log shows what is running.
    print("$", " ".join(cmd), flush=True)
    # check=True: if the command fails, raise an error and stop the whole script.
    subprocess.run(cmd, check=True)


def save_rows():
    """Copy the ablation rows from results/results.csv into results/ablation_<who>.csv.

    results.csv may also hold other rows (such as the supervised baseline); only rows whose
    tag starts with "abl_" or "bs_" are copied, so the ablation file holds ablation results only.
    """
    # Read all rows of results.csv. Each row is a list of text values.
    with open("results/results.csv") as f:
        rows = list(csv.reader(f))
    # Keep the header (rows[0]) plus every row whose second column (the tag) starts with
    # "abl_" or "bs_".
    keep = [rows[0]] + [r for r in rows[1:] if r[1].startswith(("abl_", "bs_"))]
    # Write them to this person's ablation file, replacing the old version.
    with open(f"results/ablation_{a.who}.csv", "w", newline="") as f:
        csv.writer(f).writerows(keep)


def one_run(tag, augs, batch):
    """One ablation run: pretrain with the given augmentations and batch size, then
    measure the encoder with a linear probe."""
    # Step 1: pretraining. sys.executable is the current Python program; "*augs" inserts the
    # augmentation names one by one into the command.
    run([sys.executable, "train_simclr.py", "--name", tag, "--augs", *augs, "--epochs", str(EPOCHS),
         "--subset", str(SUBSET), "--batch_size", str(batch), "--workers", str(a.workers)])
    # Step 2: linear probe on all labels, using the checkpoint saved by step 1.
    run([sys.executable, "evaluate.py", "--mode", "linear", "--ckpt", f"checkpoints/{tag}.pt",
         "--frac", "1.0", "--tag", tag])
    # Save the results after every run, so a disconnection loses at most the current run.
    save_rows()


# Run every requested augmentation setting with batch size 256 (the tag becomes e.g. abl_no_crop).
for name in a.aug:
    one_run(f"abl_{name}", AUG_SETTINGS[name], 256)
# Run every requested batch size with all augmentations on (the tag becomes e.g. bs_512).
for bs in a.bs:
    one_run(f"bs_{bs}", AUG_SETTINGS["full"], bs)
# "%s" is replaced by the person's name.
print("Done. Send results/ablation_%s.csv to Shubham." % a.who)
