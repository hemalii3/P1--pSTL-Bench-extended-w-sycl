import matplotlib
matplotlib.use("Agg")
import json, os
import numpy as np
import matplotlib.pyplot as plt

its_list = [1, 10, 100, 1000]
sizes = []

# collect data
data = {its: {} for its in its_list}
for its in its_list:
    for backend, f in [
        ('GPU-wg128', f'/home/hemali/results/foreach_gpu_its{its}.json'),
        ('TBB',       f'/home/hemali/results/foreach_tbb_its{its}.json'),
        ('GNU',       f'/home/hemali/results/foreach_gnu_its{its}.json'),
    ]:
        d = json.load(open(f))
        means = [b for b in d['benchmarks'] if 'mean' in b['name']]
        xs, ys = [], []
        for b in means:
            sz = int(b['name'].split('/')[3])
            xs.append(sz)
            ys.append(b['real_time'] / 1e6)
        data[its][backend] = (xs, ys)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

colors = {'GPU-wg128': 'red', 'TBB': 'gray', 'GNU': 'blue'}
styles = {'GPU-wg128': '-', 'TBB': '--', 'GNU': ':'}

for idx, its in enumerate(its_list):
    ax = axes[idx]
    for backend, (xs, ys) in data[its].items():
        ax.plot(xs, ys, marker='o', markersize=3,
                label=backend, color=colors[backend],
                linestyle=styles[backend], linewidth=1.5)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_title(f'for_each  KERNEL_ITS={its}', fontsize=12)
    ax.set_xlabel('Input size (elements)')
    ax.set_ylabel('Time (ms)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

plt.suptitle('for_each: Memory-bound to Compute-bound Transition', fontsize=14)
plt.tight_layout()
plt.savefig('plots_comparison/foreach_kits.png', dpi=150)
plt.close()
print("Saved plots_comparison/foreach_kits.png")
