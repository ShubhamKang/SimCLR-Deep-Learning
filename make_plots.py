"""Turn results/results.csv into the three report figures and print a summary table.

Results are grouped by `tag` (set with --tag in evaluate.py):
  supervised / simclr_full -> baseline vs SimCLR across label fractions
  abl_*                    -> augmentation ablation
  bs_*                     -> batch-size ablation
If a tag was run with several seeds, the mean is plotted.
"""
import pandas as pd, matplotlib; matplotlib.use("Agg")  # Agg = draw to files, no display needed
import matplotlib.pyplot as plt
df = pd.read_csv("results/results.csv")
df["acc"] = df.test_acc * 100  # accuracy in percent

# Figure 1: test accuracy vs. fraction of labels used (log x axis), one line per method.

m = df[df.tag.isin(["supervised", "simclr_full"])]
fig, ax = plt.subplots(figsize=(5, 3.5))
for (mode, tag), g in m.groupby(["mode", "tag"]):
    g = g.groupby("frac").acc.mean().sort_index()
    ax.plot(g.index * 100, g.values, marker="o", label=f"{tag} ({mode})")
ax.set_xscale("log"); ax.set_xlabel("Labels used (%)"); ax.set_ylabel("Test accuracy (%)"); ax.legend(); fig.tight_layout()
fig.savefig("results/labels_vs_acc.png", dpi=200)

# Figure 2: augmentation ablation, one bar per setting (linear-probe accuracy).
ab = df[df.tag.str.startswith("abl_")].groupby("tag").acc.mean().sort_values()
fig, ax = plt.subplots(figsize=(5, 3.5)); ab.plot.barh(ax=ax); ax.set_xlabel("Linear probe accuracy (%)"); fig.tight_layout()
fig.savefig("results/aug_ablation.png", dpi=200)

# Figure 3: batch size (parsed from tags such as "bs_256") vs. linear-probe accuracy.
bs = df[df.tag.str.startswith("bs_")].copy(); bs["b"] = bs.tag.str[3:].astype(int)
bs = bs.groupby("b").acc.mean()
fig, ax = plt.subplots(figsize=(5, 3.5)); ax.plot(bs.index, bs.values, marker="o"); ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size"); ax.set_ylabel("Linear probe accuracy (%)"); fig.tight_layout()
fig.savefig("results/batchsize_ablation.png", dpi=200)
# Summary table printed to the terminal; copy it into the report.
print(df.groupby(["tag", "mode", "frac"]).acc.mean().round(2))
