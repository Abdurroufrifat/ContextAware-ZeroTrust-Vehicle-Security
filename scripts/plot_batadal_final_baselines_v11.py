"""Create a compact, publication quality summary of the final BATADAL comparison."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

comparison = pd.read_csv(RESULTS / "final_baseline_comparison.csv")
episodes = pd.read_csv(RESULTS / "final_baseline_episode_comparison.csv")
methods = comparison["method"].tolist()
colors = ["#2266AA", "#CC6677", "#228833"]

fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)


def grouped_panel(ax, metrics, labels, title, ylim=(0, 1)):
    x = np.arange(len(metrics))
    width = 0.24
    for index, (method, color) in enumerate(zip(methods, colors)):
        row = comparison.loc[comparison["method"] == method].iloc[0]
        means = [row[f"{metric}_mean"] for metric in metrics]
        errors = [row[f"{metric}_sd"] if pd.notna(row[f"{metric}_sd"]) else 0 for metric in metrics]
        ax.bar(x + (index - 1) * width, means, width, yerr=errors, capsize=2.5,
               label=method, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


grouped_panel(axes[0, 0], ["precision", "recall", "f1"],
              ["Precision", "Recall", "F1"], "A  Threshold performance")
grouped_panel(axes[0, 1], ["roc_auc", "average_precision"],
              ["ROC AUC", "Average\nprecision"], "B  Ranking performance")

x = np.arange(len(methods))
episode_rows = episodes.set_index("method").loc[methods]
episode_errors = episode_rows["attack_episode_recall_sd"].fillna(0)
axes[1, 0].bar(x, episode_rows["attack_episode_recall_mean"],
               yerr=episode_errors, capsize=3, color=colors, edgecolor="white")
axes[1, 0].set_xticks(x, ["CO ACT CPS", "Isolation\nForest", "Persistence"])
axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_ylabel("Fraction of 7 attacks detected")
axes[1, 0].set_title("C  Attack episode detection", loc="left", fontweight="bold")
axes[1, 0].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[1, 0].set_axisbelow(True)

far_errors = comparison["false_alarm_rate_sd"].fillna(0)
axes[1, 1].bar(x, 100 * comparison["false_alarm_rate_mean"],
               yerr=100 * far_errors, capsize=3, color=colors, edgecolor="white")
axes[1, 1].set_xticks(x, ["CO ACT CPS", "Isolation\nForest", "Persistence"])
axes[1, 1].set_ylabel("Normal hours alarmed (%)")
axes[1, 1].set_title("D  False alarm rate", loc="left", fontweight="bold")
axes[1, 1].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[1, 1].set_axisbelow(True)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
fig.savefig(FIGURES / "batadal_final_baseline_comparison.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "batadal_final_baseline_comparison.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved comparison figures to {FIGURES}")
