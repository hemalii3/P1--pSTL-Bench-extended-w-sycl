import matplotlib
matplotlib.use("Agg")
import json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SOURCES = [
    ("results/tbb.json",         "TBB"),
    ("results/gnu.json",         "GNU"),
    ("results/sycl_cpu.json",    "SYCL-CPU"),
    ("results/sycl_wg32.json",   "SYCL-wg32"),
    ("results/sycl_wg128.json",  "SYCL-wg128"),
    ("results/sycl_wg256.json",  "SYCL-wg256"),
    ("results/sycl_wg512.json",  "SYCL-wg512"),
    ("results/sycl_wg1024.json", "SYCL-wg1024"),
]

ALGORITHMS = {"find", "for_each", "inclusive_scan", "reduce", "sort"}
OUT_DIR = "plots_comparison"

HIGHLIGHT = {
    "find":           ["TBB", "GNU", "SYCL-CPU", "SYCL-wg128", "SYCL-wg1024"],
    "for_each":       ["TBB", "GNU", "SYCL-CPU", "SYCL-wg128", "SYCL-wg32"],
    "inclusive_scan": ["TBB", "GNU", "SYCL-CPU", "SYCL-wg512", "SYCL-wg1024"],
    "reduce":         ["TBB", "GNU", "SYCL-CPU", "SYCL-wg512", "SYCL-wg1024"],
    "sort":           ["TBB", "GNU", "SYCL-CPU", "SYCL-wg256", "SYCL-wg32"],
}

LINESTYLES = {"TBB": "--", "GNU": ":", "SYCL-CPU": "-."}
LINEWIDTHS = {"TBB": 2.0, "GNU": 2.0, "SYCL-CPU": 2.0}

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
        rows.append({"label": label, "algo": algo, "size": size, "time_ns": b["real_time"]})
    return pd.DataFrame(rows)

dfs = []
for path, label in SOURCES:
    if os.path.exists(path):
        dfs.append(load_json(path, label))

df = pd.concat(dfs, ignore_index=True)
os.makedirs(OUT_DIR, exist_ok=True)

stats_rows = []
for (label, algo, size), group in df.groupby(["label", "algo", "size"]):
    times = group["time_ns"].values
    if len(times) >= 3:
        times = times[1:]
    stats_rows.append({
        "label": label, "algo": algo, "size": size,
        "median": np.median(times),
        "q25": np.percentile(times, 25),
        "q75": np.percentile(times, 75),
    })
stats = pd.DataFrame(stats_rows)

for algo in sorted(ALGORITHMS):
    subset = stats[stats["algo"] == algo]
    show = HIGHLIGHT.get(algo, [])
    subset = subset[subset["label"].isin(show)]
    if subset.empty:
        continue
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, group in subset.groupby("label"):
        group = group.sort_values("size")
        if algo == "sort":
            group = group[group["size"] >= 16]
        sizes  = group["size"].values
        median = group["median"].values / 1e6
        q25    = group["q25"].values / 1e6
        q75    = group["q75"].values / 1e6
        ls = LINESTYLES.get(label, "-")
        lw = LINEWIDTHS.get(label, 1.5)
        line, = ax.plot(sizes, median, marker="o", markersize=4,
                        label=label, linestyle=ls, linewidth=lw)
        ax.fill_between(sizes, q25, q75, alpha=0.15, color=line.get_color())
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Input size (elements)")
    ax.set_ylabel("Time (ms)  [median +/- IQR]")
    ax.set_title(f"X::{algo} - Best/Worst SYCL wg vs CPU baselines", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{algo}_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

print("Done.")
