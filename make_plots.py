"""Turn the result files into the report's figures and one combined table.

Input files (each has the columns mode, tag, frac, seed, test_acc):
  results/results.csv              supervised baseline results
  results/results_simclr_full.csv  SimCLR linear-probe and fine-tune results
  results/ablation_*.csv           augmentation (tags abl_...) and batch-size (tags bs_...) results
  results/batchsize_colab_nidhi.csv  the second batch-size run, done on Colab
Output files (in results/):
  all_results.csv         all rows merged into one table
  labels_vs_acc.png       test accuracy against the fraction of labels, one line per method
  aug_ablation.png        bar chart of the augmentation ablation
  batchsize_ablation.png  accuracy against batch size, for both batch-size runs
  pretrain_loss.png       the loss of the main pretraining run, epoch by epoch

Words used below:
  pandas (pd)  : a library for tables of data ("DataFrames"), similar to a spreadsheet.
  matplotlib   : a library for drawing charts.
  tag          : the short run name stored in each row, e.g. "abl_no_crop" or "bs_512".
"""
# glob finds files whose names match a pattern, e.g. results/ablation_*.csv.
import glob
# pandas handles tables of data.
import pandas as pd
# matplotlib draws the charts. use("Agg") makes it draw straight into image files, without
# needing a screen (useful on servers such as Kaggle).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 1. Load all result files and merge them into one table ----
# The list of files to read: the two fixed ones, plus every ablation file that exists
# (sorted() puts them in alphabetical order).
files = ["results/results.csv", "results/results_simclr_full.csv"] + sorted(glob.glob("results/ablation_*.csv"))
# Read each file that exists into a table. glob.glob(f) is empty (False) if the file is missing.
frames = [pd.read_csv(f) for f in files if glob.glob(f)]
# The second batch-size run (Colab) is kept in its own file and drawn as a separate line
# below. If a copy of its rows (tags starting with "bs_") is in results.csv, remove it here so
# the two runs are not mixed. "~" means "not".
frames[0] = frames[0][~frames[0].tag.astype(str).str.startswith("bs_")]
# pd.concat stacks all tables into one. drop_duplicates removes rows that describe the same
# run (same mode, tag, frac and seed) twice, keeping the last one read.
df = pd.concat(frames).drop_duplicates(subset=["mode", "tag", "frac", "seed"], keep="last")
# New column "acc": the accuracy as a percentage (0.5700 -> 57.0).
df["acc"] = df.test_acc * 100
# Save the merged table. index=False leaves out pandas' row numbers.
df.to_csv("results/all_results.csv", index=False)

# Chart style used by every figure: font size 10, a light grid, and no top/right border lines.
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- 2. Figure: test accuracy against the fraction of labels used ----
# Which rows to draw, and the name shown in the legend for each line.
# Each key is (mode, tag) as stored in the result files.
names = {("scratch", "supervised"): "Supervised from scratch",
         ("linear", "simclr_full"): "SimCLR + linear probe",
         ("finetune", "simclr_full"): "SimCLR + fine-tune"}
# Create an empty chart 5.5 x 3.8 inches. fig is the whole image, ax the drawing area.
fig, ax = plt.subplots(figsize=(5.5, 3.8))
# One line per method.
for (mode, tag), label in names.items():
    # Keep the rows of this method, group them by label fraction, and take the mean accuracy
    # (useful if a setting was run with several seeds). sort_index() orders by fraction.
    g = df[(df["mode"] == mode) & (df.tag == tag)].groupby("frac").acc.mean().sort_index()
    # Draw only if there are rows. x = fraction in percent, y = accuracy; marker="o" draws dots.
    if len(g):
        ax.plot(g.index * 100, g.values, marker="o", label=label)
# Logarithmic x axis: 1%, 10% and 100% are equally spaced, so the small fractions are readable.
ax.set_xscale("log")
# Put tick marks exactly at the four fractions, labelled with "%".
ax.set_xticks([1, 10, 20, 100]); ax.set_xticklabels(["1%", "10%", "20%", "100%"])
# Axis titles.
ax.set_xlabel("Training labels used (of 5,000)"); ax.set_ylabel("Test accuracy (%)")
# Legend without a box, tidy the layout, and save the image at 200 dots per inch.
ax.legend(frameon=False); fig.tight_layout(); fig.savefig("results/labels_vs_acc.png", dpi=200)

# ---- 3. Figure: augmentation ablation (bar chart) ----
# Rows whose tag starts with "abl_", mean accuracy per setting, sorted from lowest to highest.
ab = df[df.tag.str.startswith("abl_")].groupby("tag").acc.mean().sort_values()
if len(ab):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    # One colour per bar: red for the "full" setting (the reference), grey for the others.
    colors = ["#c0392b" if t == "abl_full" else "#7f8c8d" for t in ab.index]
    # Horizontal bars. t[4:] removes the first 4 characters ("abl_") from each name.
    ax.barh([t[4:] for t in ab.index], ab.values, color=colors)
    # Write each bar's value just to the right of its end (v + 0.5), with one decimal.
    for y, v in enumerate(ab.values):
        ax.text(v + 0.5, y, f"{v:.1f}", va="center", fontsize=8)
    # x-axis title, and remove the horizontal grid lines, which are not needed for bars.
    ax.set_xlabel("Linear-probe test accuracy (%)"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig("results/aug_ablation.png", dpi=200)

# ---- 4. Figure: batch-size ablation, two runs with identical settings ----
# Run 1 (Kaggle): the rows whose tag starts with "bs_" (64, 128, 512) from the merged table.
# .copy() makes an independent copy so we can add a column safely.
bs = df[df.tag.str.startswith("bs_")].copy()
# New column "b" = the batch size, read from the tag: "bs_512"[3:] = "512" -> 512.
bs["b"] = bs.tag.str[3:].astype(int)
# The batch-256 point: the "full" augmentation run used exactly the same settings as a
# batch-256 run (all augmentations, batch 256, same schedule), so it is reused here.
full256 = df[df.tag == "abl_full"].assign(b=256)
# Combine, then mean accuracy per batch size, ordered by batch size.
run1 = pd.concat([bs, full256]).groupby("b").acc.mean().sort_index()
fig, ax = plt.subplots(figsize=(5.5, 3.8))
# Solid line with dots for run 1.
if len(run1):
    ax.plot(run1.index, run1.values, marker="o", label="Run 1 (Kaggle T4)")
# Run 2 (Colab), from its own file, drawn as a dashed line with squares.
if glob.glob("results/batchsize_colab_nidhi.csv"):
    c = pd.read_csv("results/batchsize_colab_nidhi.csv")
    c["b"] = c.tag.str[3:].astype(int)
    # This file stores accuracy as a fraction, so multiply by 100 for percent.
    run2 = (c.groupby("b").test_acc.mean() * 100).sort_index()
    ax.plot(run2.index, run2.values, marker="s", linestyle="--", label="Run 2 (Colab T4)")
# Base-2 logarithmic x axis, so 64, 128, 256 and 512 are equally spaced (each doubles).
ax.set_xscale("log", base=2); ax.set_xticks([64, 128, 256, 512]); ax.set_xticklabels([64, 128, 256, 512])
ax.set_xlabel("Batch size"); ax.set_ylabel("Linear-probe test accuracy (%)")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig("results/batchsize_ablation.png", dpi=200)

# ---- 5. Print a summary table ----
# Mean accuracy for every (tag, mode, fraction), rounded to 1 decimal, printed as text.
print(df.groupby(["tag", "mode", "frac"]).acc.mean().round(1).to_string())

# ---- 6. Figure: loss of the main pretraining run ----
# os is needed to check whether the log file exists.
import os
if os.path.exists("results/logs/pretrain_full.csv"):
    # The log has one row per epoch: epoch, loss, seconds.
    lg = pd.read_csv("results/logs/pretrain_full.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    # Loss (y) against epoch (x).
    ax.plot(lg.epoch, lg.loss)
    ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent loss")
    fig.tight_layout(); fig.savefig("results/pretrain_loss.png", dpi=200)
