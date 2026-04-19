import matplotlib
matplotlib.use("Agg")
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
SOURCES = [
    ("results/tbb.json",         "TBB"),
    ("results/gnu.json",         "GNU"),
    ("results/sycl_wg32.json",   "SYCL-wg32"),
    ("results/sycl_wg64.json",   "SYCL-wg64"),
    ("results/sycl_wg128.json",  "SYCL-wg128"),
    ("results/sycl_wg256.json",  "SYCL-wg256"),
    ("results/sycl_wg512.json",  "SYCL-wg512"),
    ("results/sycl_wg1024.json", "SYCL-wg1024"),
]

ALGORITHMS = {"find", "for_each", "inclusive_scan", "reduce"}
OUT_DIR = "plots_stats"
CV_WARN_THRESHOLD = 0.05  


def load_json(filepath, label):
    """
    Load a Google Benchmark JSON file.
    Handles both single-run and multi-repetition outputs.
    For multi-rep runs, skips the aggregate rows (mean/median/stddev)
    and keeps only the per-repetition rows so we can compute our own stats.
    """
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
            "label":   label,
            "algo":    algo,
            "size":    size,
            "time_ns": b["real_time"],
        })

    return pd.DataFrame(rows)


dfs = []
for path, label in SOURCES:
    if os.path.exists(path):
        dfs.append(load_json(path, label))
    else:
        print(f"Skipping {path} (not found)")

if not dfs:
    print("No result files found.")
    exit(1)

df = pd.concat(dfs, ignore_index=True)
os.makedirs(OUT_DIR, exist_ok=True)

stats_rows = []

for (label, algo, size), group in df.groupby(["label", "algo", "size"]):
    times = group["time_ns"].values

    if len(times) >= 3:
        times = times[1:]

    median = np.median(times)
    mean   = np.mean(times)
    std    = np.std(times, ddof=1) if len(times) > 1 else 0.0
    q25    = np.percentile(times, 25)
    q75    = np.percentile(times, 75)
    cv     = (std / mean) if mean > 0 else 0.0

    if cv > CV_WARN_THRESHOLD and len(times) > 1:
        print(f"  [WARN] High CV={cv:.2%}  label={label}  algo={algo}  size={size}")

    stats_rows.append({
        "label":  label,
        "algo":   algo,
        "size":   size,
        "median": median,
        "mean":   mean,
        "std":    std,
        "q25":    q25,
        "q75":    q75,
        "count":  len(times),
        "cv":     cv,
    })

stats = pd.DataFrame(stats_rows)
csv_path = os.path.join(OUT_DIR, "summary_stats.csv")
stats.to_csv(csv_path, index=False)
print(f"Saved {csv_path}")

for algo in sorted(ALGORITHMS):
    subset = stats[stats["algo"] == algo]
    if subset.empty:
        continue

    fig, ax = plt.subplots(figsize=(11, 5))

    for label, group in subset.groupby("label"):
        group = group.sort_values("size")
        sizes  = group["size"].values
        median = group["median"].values / 1e6   # ns → ms
        q25    = group["q25"].values   / 1e6
        q75    = group["q75"].values   / 1e6

        line, = ax.plot(sizes, median, marker="o", markersize=4, label=label)
        ax.fill_between(sizes, q25, q75, alpha=0.15, color=line.get_color())

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Input size (elements)")
    ax.set_ylabel("Time (ms)  [median ± IQR]")
    ax.set_title(f"X::{algo} - Problem Scaling  (median + IQR shading)", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{algo}_stats.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
print("finished.")
