"""Turn every results CSV into the report figures and one combined table.

Inputs (all with columns mode, tag, frac, seed, test_acc):
  results/results.csv              supervised baseline (Sandhya)
  results/results_simclr_full.csv  SimCLR linear probe + fine-tune (50-epoch checkpoint)
  results/ablation_*.csv           augmentation (abl_*) and batch-size (bs_*) ablations
Outputs (in results/):
  all_results.csv          every row merged, duplicates removed
  labels_vs_acc.png        test accuracy vs fraction of labels, one line per method
  aug_ablation.png         linear-probe accuracy for each augmentation setting
  batchsize_ablation.png   linear-probe accuracy vs batch size
If a setting was run with several seeds, the mean is used.
"""
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw straight to files; no screen needed
import matplotlib.pyplot as plt

# ---- 1. load and merge every CSV ----
files = ["results/results.csv", "results/results_simclr_full.csv"] + sorted(glob.glob("results/ablation_*.csv"))
frames = [pd.read_csv(f) for f in files if glob.glob(f)]
df = pd.concat(frames).drop_duplicates(subset=["mode", "tag", "frac", "seed"], keep="last")
df["acc"] = df.test_acc * 100  # accuracy in percent
df.to_csv("results/all_results.csv", index=False)

# Shared style: quiet grid, readable labels.
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- 2. Figure 1: accuracy vs labels used ----
names = {("scratch", "supervised"): "Supervised from scratch",
         ("linear", "simclr_full"): "SimCLR + linear probe",
         ("finetune", "simclr_full"): "SimCLR + fine-tune"}
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for (mode, tag), label in names.items():
    g = df[(df["mode"] == mode) & (df.tag == tag)].groupby("frac").acc.mean().sort_index()
    if len(g):
        ax.plot(g.index * 100, g.values, marker="o", label=label)
ax.set_xscale("log")
ax.set_xticks([1, 10, 20, 100]); ax.set_xticklabels(["1%", "10%", "20%", "100%"])
ax.set_xlabel("Training labels used (of 5,000)"); ax.set_ylabel("Test accuracy (%)")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig("results/labels_vs_acc.png", dpi=200)

# ---- 3. Figure 2: augmentation ablation ----
ab = df[df.tag.str.startswith("abl_")].groupby("tag").acc.mean().sort_values()
if len(ab):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    colors = ["#c0392b" if t == "abl_full" else "#7f8c8d" for t in ab.index]  # highlight the full set
    ax.barh([t[4:] for t in ab.index], ab.values, color=colors)
    for y, v in enumerate(ab.values):
        ax.text(v + 0.5, y, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Linear-probe test accuracy (%)"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig("results/aug_ablation.png", dpi=200)

# ---- 4. Figure 3: batch size (bs_* runs, plus abl_full as the batch-256 point) ----
bs = df[df.tag.str.startswith("bs_")].copy()
bs["b"] = bs.tag.str[3:].astype(int)
full256 = df[df.tag == "abl_full"].assign(b=256)  # identical settings to a bs_256 run
bs = pd.concat([bs, full256]).groupby("b").acc.mean().sort_index()
if len(bs):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(bs.index, bs.values, marker="o")
    ax.set_xscale("log", base=2); ax.set_xticks(bs.index); ax.set_xticklabels(bs.index)
    ax.set_xlabel("Batch size"); ax.set_ylabel("Linear-probe test accuracy (%)")
    fig.tight_layout(); fig.savefig("results/batchsize_ablation.png", dpi=200)

# ---- 5. summary table for the report ----
print(df.groupby(["tag", "mode", "frac"]).acc.mean().round(1).to_string())
