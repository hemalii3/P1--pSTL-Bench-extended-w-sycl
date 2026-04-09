#!/usr/bin/env bash
# run_analysis.sh — Analysis pipeline for pSTL-Bench

echo "========================================"
echo " pSTL-Bench Analysis Pipeline"
echo "========================================"

python3 -c "import pandas, matplotlib, numpy, scipy" 2>/dev/null || {
    echo "[ERROR] Missing Python dependencies."
    exit 1
}

echo ""
echo "[1/2] Statistical plots: median + IQR..."
python3 plot_results_stats.py

echo ""
echo "[2/2] Work-group size analysis..."
python3 compare_wg_sizes.py

echo ""
echo "========================================"
echo " Done."
echo "   plots_stats/      — median + IQR plots"
echo "   plots_wg_compare/ — wg_size overlays and heatmaps"
echo "========================================"
