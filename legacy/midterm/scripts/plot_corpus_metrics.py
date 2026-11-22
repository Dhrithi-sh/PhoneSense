#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot visual figures for Dhvani Regional News Crawler & Corpus (Part 1 - Riya).

Generates report-ready PNG figures in `documentation/figures/`:
1. `riya-dedup-precision-recall.png`  - Deduplication calibration & precision/recall curves
2. `riya-corpus-distribution.png`     - Source balance and geographic coverage across 5,000 articles
3. `riya-recrawl-bandwidth.png`       - HTTP 304 conditional request savings and publication velocities
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure clean styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.titlesize": 13,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "documentation" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_deduplication_metrics():
    """Figure 1: Benchmark calibration curves on 100 labeled Hindi pairs."""
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    precision = [1.000, 1.000, 1.000, 1.000, 1.000]
    recall = [1.000, 1.000, 0.950, 0.650, 0.425]
    f1 = [1.000, 1.000, 0.974, 0.788, 0.597]

    tp = [40, 40, 38, 26, 17]
    fn = [0, 0, 2, 14, 23]
    tn = [60, 60, 60, 60, 60]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot 1: Performance curves
    ax1.plot(thresholds, precision, "s-", color="#2b5c8f", label="Precision", linewidth=2.2, markersize=6)
    ax1.plot(thresholds, recall, "o-", color="#d95f02", label="Recall", linewidth=2.2, markersize=6)
    ax1.plot(thresholds, f1, "^-", color="#2ca02c", label="F1 Score", linewidth=2.2, markersize=6)

    ax1.axvline(x=0.70, color="#d62728", linestyle="--", alpha=0.85, label="Optimal Standard ($J=0.70$)")
    ax1.scatter([0.70], [0.974], color="#d62728", s=100, zorder=5)
    ax1.annotate(
        "Optimal Sweet Spot\nP=1.00, R=0.95, F1=0.974",
        xy=(0.70, 0.974),
        xytext=(0.71, 0.88),
        arrowprops=dict(facecolor="#333", arrowstyle="->", lw=1.2),
        fontsize=9,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf6e2", ec="#b58900", alpha=0.9),
    )

    ax1.set_xlabel("Jaccard Similarity Threshold ($J$)")
    ax1.set_ylabel("Metric Score")
    ax1.set_title("Deduplication Performance vs. Jaccard Cutoff")
    ax1.set_ylim(0.35, 1.05)
    ax1.set_xlim(0.58, 0.82)
    ax1.set_xticks(thresholds)
    ax1.legend(loc="lower left", framealpha=0.9)

    # Subplot 2: Classification counts
    bar_width = 0.012
    ax2.bar([t - bar_width for t in thresholds], tp, width=bar_width, label="True Positives (Syndications Caught)", color="#2ca02c")
    ax2.bar([t for t in thresholds], fn, width=bar_width, label="False Negatives (Missed Copies)", color="#d95f02")
    ax2.bar([t + bar_width for t in thresholds], tn, width=bar_width, label="True Negatives (Distinct Kept)", color="#1f77b4")

    ax2.axvline(x=0.70, color="#d62728", linestyle="--", alpha=0.85)
    ax2.set_xlabel("Jaccard Similarity Threshold ($J$)")
    ax2.set_ylabel("Pair Count (out of 100)")
    ax2.set_title("Classification Yield (0 False Positives Across All Cutoffs)")
    ax2.set_xticks(thresholds)
    ax2.set_ylim(0, 70)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    out_file = FIGURES_DIR / "riya-dedup-precision-recall.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


def plot_corpus_distribution():
    """Figure 2: Source balance and geographic representation across 5,000 articles."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Panel 1: Source distribution
    sources = ["Live Hindustan", "Amar Ujala", "Dainik Jagran", "Aaj Tak", "Navbharat Times"]
    counts = [1041, 1038, 1036, 1002, 884]
    agencies = [70, 43, 46, 90, 44]

    y_pos = np.arange(len(sources))
    ax1.barh(y_pos, counts, color="#3182bd", height=0.55, label="Standard Articles", edgecolor="#1c4e80")
    ax1.barh(y_pos, agencies, color="#fd8d3c", height=0.55, label="Syndicated Wire Feeds (agency_flag)", edgecolor="#d94701")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sources, fontweight="medium")
    ax1.invert_yaxis()
    ax1.set_xlabel("Articles Crawled")
    ax1.set_title("Source Balance & Wire Agency Distribution (5,001 Articles)")
    ax1.set_xlim(0, 1250)

    for i, (c, a) in enumerate(zip(counts, agencies)):
        ax1.text(c + 15, i, f"{c:,} ({c/5001*100:.1f}%)", va="center", fontsize=9, fontweight="semibold")

    ax1.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

    # Panel 2: Geographic coverage (top states)
    states = [
        "Uttar Pradesh", "Bihar", "Delhi", "Uttarakhand",
        "Madhya Pradesh", "Haryana", "Jharkhand", "Punjab",
        "Himachal Pradesh", "Rajasthan", "Chhattisgarh"
    ]
    state_counts = [1152, 323, 272, 178, 175, 130, 129, 110, 110, 97, 56]
    y_states = np.arange(len(states))

    ax2.barh(y_states, state_counts, color="#41ab5d", height=0.6, edgecolor="#238b45")
    ax2.set_yticks(y_states)
    ax2.set_yticklabels(states)
    ax2.invert_yaxis()
    ax2.set_xlabel("Articles Tagged")
    ax2.set_title("Geographic Representation across 11 Hindi-Belt States\n(2,732 Tagged Articles across 120+ Cities)")
    ax2.set_xlim(0, 1350)

    for i, sc in enumerate(state_counts):
        ax2.text(sc + 15, i, f"{sc:,} ({sc/2732*100:.1f}%)", va="center", fontsize=8.5)

    plt.tight_layout()
    out_file = FIGURES_DIR / "riya-corpus-distribution.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


def plot_recrawl_bandwidth():
    """Figure 3: Adaptive recrawling bandwidth savings and velocities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Panel 1: Bandwidth reduction
    labels = ["HTTP 200 OK\n(Fresh Sitemaps - 70)", "HTTP 304 Not Modified\n(Cached Savings - 10)"]
    sizes = [70, 10]
    colors = ["#2b5c8f", "#2ca02c"]
    explode = (0, 0.1)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140, textprops=dict(color="#222", fontsize=9.5),
        wedgeprops=dict(edgecolor="white", width=0.65)
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(10)

    ax1.set_title("Adaptive Recrawling Bandwidth Efficiency\n(12.5% Conditional HTTP 304 Savings)")

    # Panel 2: Measured Publication Velocities
    sources = ["Amar Ujala", "Dainik Jagran", "Aaj Tak", "Live Hindustan", "Navbharat Times"]
    velocities = [54.52, 51.68, 25.41, 21.98, 20.91]
    y_pos = np.arange(len(sources))

    ax2.barh(y_pos, velocities, color="#6baed6", height=0.55, edgecolor="#3182bd")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sources)
    ax2.invert_yaxis()
    ax2.set_xlabel("Estimated Publication Velocity ($\\lambda_s$ URLs/hour)")
    ax2.set_title("Domain Publication Velocity ($\\lambda_s$ EWMA)")
    ax2.set_xlim(0, 65)

    for i, v in enumerate(velocities):
        ax2.text(v + 1.0, i, f"{v:.1f} / h", va="center", fontsize=9, fontweight="semibold")

    plt.tight_layout()
    out_file = FIGURES_DIR / "riya-recrawl-bandwidth.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


def main():
    print("Generating visual figures for crawler and corpus documentation...")
    plot_deduplication_metrics()
    plot_corpus_distribution()
    plot_recrawl_bandwidth()
    print("All figures successfully created in documentation/figures/!")


if __name__ == "__main__":
    main()
