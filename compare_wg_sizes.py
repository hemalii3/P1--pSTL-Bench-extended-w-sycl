import matplotlib
matplotlib.use("Agg")
import json
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal

WG_SIZES = [32, 64, 128, 256, 512, 1024]
SOURCES = [(f"results/sycl_wg{wg}.json", f"wg{wg}") for wg in WG_SIZES]
ALGORITHMS = {"find", "for_each", "inclusive_scan", "reduce", "sort"}
OUT_DIR = "plots_wg_compare"


def load_json(filepath, label):
    with open(filepath) as f:
        data = json.load(f)
    rows = []
    for b in data["benchmarks"]:
        if b.get("run_type") == "aggregate":
            continue
        parts = b["name"].split("/")
        if len(parts) < 4:
            continue
        algo = parts[1]
        for ns in ("std::", "sycl::", "gnu::", "tbb::", "hpx::"):
            algo = algo.replace(ns, "")
        if algo not in ALGORITHMS:
            continue
        try:
            size = int(parts[3])
        except ValueError:
            continue
        rows.append({
            "wg":      label,
            "algo":    algo,
            "size":    size,
            "time_ns": b["real_time"],
        })
    return pd.DataFrame(rows)


dfs = []
loaded_wgs = []
for path, label in SOURCES:
    if os.path.exists(path):
        dfs.append(load_json(path, label))
        loaded_wgs.append(label)
    else:
        print(f"Skipping {path} (not found)")

if not dfs:
    print("No SYCL result files found.")
    exit(1)

df = pd.concat(dfs, ignore_index=True)
os.makedirs(OUT_DIR, exist_ok=True)

stats = (
    df.groupby(["wg", "algo", "size"])["time_ns"]
    .agg(median="median", mean="mean", count="count")
    .reset_index()
)

pairwise_rows = []

for algo in sorted(ALGORITHMS):
    sub = stats[stats["algo"] == algo]
    sizes = sorted(sub["size"].unique())

    for wg_a, wg_b in itertools.combinations(loaded_wgs, 2):
        rel_diffs = []
        for size in sizes:
            row_a = sub[(sub["wg"] == wg_a) & (sub["size"] == size)]
            row_b = sub[(sub["wg"] == wg_b) & (sub["size"] == size)]
            if row_a.empty or row_b.empty:
                continue
            m_a = row_a["median"].values[0]
            m_b = row_b["median"].values[0]
            denom = (m_a + m_b) / 2.0
            if denom > 0:
                rel_diffs.append(abs(m_a - m_b) / denom * 100.0)

        if rel_diffs:
            pairwise_rows.append({
                "algo":            algo,
                "wg_a":            wg_a,
                "wg_b":            wg_b,
                "mean_rel_diff_%": np.mean(rel_diffs),
                "max_rel_diff_%":  np.max(rel_diffs),
            })

pairwise_df = pd.DataFrame(pairwise_rows)
pairwise_csv = os.path.join(OUT_DIR, "wg_pairwise_diff.csv")
pairwise_df.to_csv(pairwise_csv, index=False)
print(f"Saved {pairwise_csv}")

kw_rows = []

for algo in sorted(ALGORITHMS):
    sub = df[df["algo"] == algo]
    sizes = sorted(sub["size"].unique())

    for size in sizes:
        groups = []
        for wg in loaded_wgs:
            times = sub[(sub["wg"] == wg) & (sub["size"] == size)]["time_ns"].values
            if len(times) > 0:
                groups.append(times)

        if len(groups) < 2:
            continue

        if all(len(g) == 1 for g in groups):
            pass

        try:
            h_stat, p_val = kruskal(*groups)
        except ValueError:
            continue

        kw_rows.append({
            "algo":        algo,
            "size":        size,
            "H_stat":      round(h_stat, 4),
            "p_value":     round(p_val, 6),
            "significant": p_val < 0.05,
        })

kw_df = pd.DataFrame(kw_rows)
kw_csv = os.path.join(OUT_DIR, "wg_kruskal_wallis.csv")
kw_df.to_csv(kw_csv, index=False)
print(f"Saved {kw_csv}")

sig_count = kw_df["significant"].sum() if not kw_df.empty else 0
total     = len(kw_df)
# KW print suppressed

COLORS = plt.cm.tab10.colors

for algo in sorted(ALGORITHMS):
    sub = stats[stats["algo"] == algo]
    if sub.empty:
        continue

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, wg in enumerate(loaded_wgs):
        group = sub[sub["wg"] == wg].sort_values("size")
        ax.plot(
            group["size"], group["median"] / 1e6,
            marker="o", markersize=4,
            color=COLORS[i % len(COLORS)],
            label=wg,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Input size (elements)")
    ax.set_ylabel("Median time (ms)")
    ax.set_title(f"SYCL wg_size comparison - sycl::{algo}", fontsize=13)
    ax.legend(title="work-group size", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"wg_overlay_{algo}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

if not pairwise_df.empty:
    for algo in sorted(ALGORITHMS):
        sub = pairwise_df[pairwise_df["algo"] == algo]
        if sub.empty:
            continue

        wgs = loaded_wgs
        n = len(wgs)
        idx = {w: i for i, w in enumerate(wgs)}
        matrix = np.zeros((n, n))

        for _, row in sub.iterrows():
            i, j = idx[row["wg_a"]], idx[row["wg_b"]]
            matrix[i, j] = row["mean_rel_diff_%"]
            matrix[j, i] = row["mean_rel_diff_%"]

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0)
        plt.colorbar(im, ax=ax, label="Mean relative diff (%)")
        ax.set_xticks(range(n)); ax.set_xticklabels(wgs, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(wgs)
        ax.set_title(f"wg_size pairwise diff - sycl::{algo}", fontsize=12)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i,j]:.1f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if matrix[i,j] < matrix.max()*0.6 else "white")

        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"wg_heatmap_{algo}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")

print("Done.")
