# -*- coding: utf-8 -*-
"""Append human-study figures to make_figures.py outputs.

Creates:
  fig_human_forest.pdf  -- per-condition mean annoyance with SD (51 observers)
                          and the paired T-E effect on a severity axis.
Reads results/human_study_analysis.json (produced by study_platform/analysis.py).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

with open(os.path.join(RES, "human_study_analysis.json"), encoding="utf-8") as f:
    hs = json.load(f)

h1 = hs["h1_paired_T_minus_E"]
ct = hs["conditions"]
conds = ["E", "T", "X", "N", "C"]
cond_labels = {"E": "E (edge-aligned)", "T": "T (interior texture)",
               "X": "X (mixed)", "N": "N (natural)", "C": "C (control)"}

fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.1),
                         gridspec_kw={"width_ratios": [1.15, 1.0]})

# (left) per-condition means with SD error bars
ax = axes[0]
means = [ct["condition_means"][c] for c in conds]
sds = [ct["condition_sd"][c] for c in conds]
colors = ["#6b7280", "#111827", "#9ca3af", "#374151", "#d1d5db"]
ax.bar(range(len(conds)), means, yerr=sds, color=colors, width=0.62,
       capsize=2.5, error_kw=dict(lw=1.0))
for i, m in enumerate(means):
    ax.text(i, m + sds[i] + 1.4, f"{m:.1f}", ha="center", fontsize=6.8)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels(conds)
ax.set_ylabel("mean annoyance (0--100)", fontsize=7.5)
ax.set_ylim(0, 76)
ax.set_title(f"Per-condition annoyance ({ct['n_observers']} observers)",
             loc="left", fontsize=8.2)

# (right) paired T-E effect with bootstrap CI
ax = axes[1]
mean, lo, hi = h1["mean"], h1["boot_ci95"][0], h1["boot_ci95"][1]
ax.errorbar([mean], [0], xerr=[[mean - lo], [hi - mean]], fmt="o",
            ms=5, lw=1.4, capsize=2.8, color="#111827", zorder=3)
ax.axvline(0, color="#111827", lw=0.7, ls="--", zorder=1)
ax.annotate(f"$p={h1['wilcoxon_p']:.2g}$, $d_z={h1['dz']:+.2f}$",
            (hi, 0), xytext=(6, 0), textcoords="offset points",
            fontsize=7, va="center", color="#1f2937",
            annotation_clip=False)
ax.set_yticks([])
ax.set_xlabel("per-observer paired effect  T$-$E", fontsize=7.5)
ax.set_xlim(-2, 11)
ax.set_title("Interior-texture vs. edge-aligned", loc="left", fontsize=8.2)

fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(HERE, "fig_human_study.pdf"),
            bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("figure written: fig_human_study.pdf")
