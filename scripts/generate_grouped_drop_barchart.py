"""
generate_grouped_drop_barchart.py
=================================
Generates a publication-grade Grouped Bar Chart with both Positive and Negative
Y-axis, comparing relative percentage changes in RMSE across the 3 stress suites
(Suite 2, Suite 3, Suite 4) relative to the Suite 1 (IID) baseline.

- Positive Y-axis: Error Increase / Performance Degradation (e.g. +108% for XGBoost)
- Negative Y-axis: Error Reduction / Performance Improvement (e.g. -23% for Thermodynamic)
- Horizontal line at y=0 indicates zero shift from IID baseline.
"""

import os
import shutil
import tempfile
from pathlib import Path

# Matplotlib configuration for headless environment
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_copolymers")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": ":",
})

BASE_DIR = Path("C:/Projects/copolymers")
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("C:/Users/Suhani  Sharma/.gemini/antigravity/brain/716101ab-b1de-4a5f-b2b4-2f12d72f1910")

def main():
    csv_path = RESULTS_DIR / "percentage_drop_matrix.csv"
    df = pd.read_csv(csv_path)

    models = df["Paradigm"].tolist()
    s2_vals = [float(x.replace("%", "").replace("+", "").strip()) for x in df["% Drop Suite 2"]]
    s3_vals = [float(x.replace("%", "").replace("+", "").strip()) for x in df["% Drop Suite 3"]]
    s4_vals = [float(x.replace("%", "").replace("+", "").strip()) for x in df["% Drop Suite 4"]]

    x = np.arange(len(models))
    width = 0.26  # Width of each grouped bar

    fig, ax = plt.subplots(figsize=(16, 7.5))

    # Background shaded zones for positive/negative guidance
    ax.axhspan(0, 130, color="#fbb4ae", alpha=0.15, zorder=0)
    ax.axhspan(-45, 0, color="#ccebc5", alpha=0.18, zorder=0)

    # Highlight zero line
    ax.axhline(0, color="black", linewidth=1.4, linestyle="-", zorder=3)

    # Colors for the 3 stress suites
    c_s2 = "#e6ab02"  # Amber / Core-to-Tail
    c_s3 = "#d95f02"  # Deep Crimson / One-Sided Extrapolation
    c_s4 = "#7570b3"  # Purple / Equimolar Gap

    rects1 = ax.bar(x - width, s2_vals, width, label="Suite 2: Core-to-Tail Extrapolation (f < 0.25 or f > 0.75)",
                    color=c_s2, edgecolor="black", linewidth=0.8, zorder=4)
    rects2 = ax.bar(x, s3_vals, width, label="Suite 3: One-Sided Extrapolation (f > 0.50 held out)",
                    color=c_s3, edgecolor="black", linewidth=0.8, zorder=4)
    rects3 = ax.bar(x + width, s4_vals, width, label="Suite 4: Equimolar Gap Interpolation (0.25 <= f <= 0.75 held out)",
                    color=c_s4, edgecolor="black", linewidth=0.8, zorder=4)

    # Annotate values on each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            va = "bottom" if height >= 0 else "top"
            offset = 2.0 if height >= 0 else -2.5
            color = "#8c0000" if height > 50 else ("#005500" if height < -15 else "black")
            weight = "bold" if abs(height) > 30 else "normal"
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + offset,
                f"{height:+.1f}%",
                ha="center",
                va=va,
                fontsize=8.5,
                fontweight=weight,
                color=color,
                zorder=5,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Zone annotations
    ax.text(0.015, 0.94, "▲ Performance Degradation / Error Increase (%)",
            transform=ax.transAxes, fontsize=11, fontweight="bold", color="#a50f15")
    ax.text(0.015, 0.08, "▼ Performance Improvement / Error Reduction (%)",
            transform=ax.transAxes, fontsize=11, fontweight="bold", color="#006d2c")

    ax.set_ylabel("Relative RMSE Change Compared to Suite 1 IID (%)", fontsize=12, fontweight="bold")
    ax.set_title("Relative Performance Shift Under Composition Stress Suites (Baseline = Suite 1 Random Holdout)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=11, fontweight="bold")
    ax.set_ylim(-45, 125)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    plt.tight_layout()
    out_path = FIG_DIR / "fig6_grouped_drop_barchart.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Mirror to artifact directory
    shutil.copy(out_path, ARTIFACT_DIR / "fig6_grouped_drop_barchart.png")
    print(f"Successfully generated grouped bar chart: {out_path}")
    print(f"Copied to artifact directory: {ARTIFACT_DIR / 'fig6_grouped_drop_barchart.png'}")

if __name__ == "__main__":
    main()
