import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
df = pd.read_csv("results/results.csv")
df["acc"] = df.test_acc * 100

m = df[df.tag.isin(["supervised", "simclr_full"])]
fig, ax = plt.subplots(figsize=(5, 3.5))
for (mode, tag), g in m.groupby(["mode", "tag"]):
    g = g.groupby("frac").acc.mean().sort_index()
    ax.plot(g.index * 100, g.values, marker="o", label=f"{tag} ({mode})")
ax.set_xscale("log"); ax.set_xlabel("Labels used (%)"); ax.set_ylabel("Test accuracy (%)"); ax.legend(); fig.tight_layout()
fig.savefig("results/labels_vs_acc.png", dpi=200)

ab = df[df.tag.str.startswith("abl_")].groupby("tag").acc.mean().sort_values()
fig, ax = plt.subplots(figsize=(5, 3.5)); ab.plot.barh(ax=ax); ax.set_xlabel("Linear probe accuracy (%)"); fig.tight_layout()
fig.savefig("results/aug_ablation.png", dpi=200)

bs = df[df.tag.str.startswith("bs_")].copy(); bs["b"] = bs.tag.str[3:].astype(int)
bs = bs.groupby("b").acc.mean()
fig, ax = plt.subplots(figsize=(5, 3.5)); ax.plot(bs.index, bs.values, marker="o"); ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size"); ax.set_ylabel("Linear probe accuracy (%)"); fig.tight_layout()
fig.savefig("results/batchsize_ablation.png", dpi=200)
print(df.groupby(["tag", "mode", "frac"]).acc.mean().round(2))
