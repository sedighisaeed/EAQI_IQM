"""Generate publication figures from the executed study JSONs (real data only).

Outputs (vector PDF, single-column width friendly):
  fig_location_forest.pdf  -- paired T-E effects with 95% bootstrap CIs (study 1)
  fig_dose_response.pdf    -- SRCC(clip fraction, score) per measure (study 2)
  fig_koniq_srcc.pdf       -- KonIQ-10k SRCC: E-AQI vs training-free baselines
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "..", "results"))

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

# ---------------------------------------------------------------- study 1
with open(os.path.join(RES, "clipping_location_study.json"), encoding="utf-8") as f:
    loc = json.load(f)
agg = loc["aggregate"]
order = ["brisque", "niqe", "piqe", "topiq_nr"]
labels = {"brisque": "BRISQUE", "niqe": "NIQE", "piqe": "PIQE",
          "topiq_nr": "TOPIQ-NR"}
# For lower-better measures, more severe = higher raw score, so T-E on the raw
# scale already follows the "severity" sign. TOPIQ-NR is higher-better, so its
# severity-scale difference is -(T-E). We plot all measures on one severity
# axis (positive = interior-texture clipping rated more severe).
fig, ax = plt.subplots(figsize=(3.35, 2.05))
for i, m in enumerate(order):
    a = agg[m]
    mean = a["mean_diff"]
    lo, hi = a["boot_ci95"]
    if m == "topiq_nr":
        mean, lo, hi = -mean, -hi, -lo
    p = a["wilcoxon_p"]
    sig = p < 0.05
    ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]],
                fmt="o", ms=4.5, lw=1.4, capsize=2.5,
                color="#1f2937" if sig else "#9ca3af",
                mfc="#1f2937" if sig else "white", zorder=3)
    ax.annotate(f"$p={p:.3g}$", (hi, i), xytext=(4, -2),
                textcoords="offset points", fontsize=6.5,
                color="#1f2937" if sig else "#6b7280",
                annotation_clip=False)
ax.axvline(0, color="#111827", lw=0.7, ls="--", zorder=1)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([labels[m] for m in order])
ax.invert_yaxis()
ax.set_xlabel("Paired effect (severity scale): interior texture $-$ edge\n"
              "(native units; TOPIQ-NR sign-flipped), 95% bootstrap CI",
              fontsize=7.2)
ax.set_title("Clipping-location effect (168 pairs)", loc="left", fontsize=8.2)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(HERE, "fig_location_forest.pdf"),
            bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---------------------------------------------------------------- study 2
with open(os.path.join(RES, "enhancement_clip_response.json"), encoding="utf-8") as f:
    dose = json.load(f)
rows = dose["rows"]
fig, axes = plt.subplots(1, 4, figsize=(7.05, 1.75), sharey=False)
for ax, m in zip(axes, order):
    cf = [r["clip_frac"] * 100 for r in rows]
    sc = [r["scores"][m] for r in rows]
    ax.scatter(cf, sc, s=4, alpha=0.45, color="#374151", linewidths=0)
    s = [d for d in dose["srcc_clipfrac_vs_score"] if d["metric"] == m][0]
    ax.set_title(f"{labels[m]}\n$\\rho={s['srcc_clipfrac_vs_score']:+.3f}$,"
                 f" $p={s['p_value']:.2g}$", fontsize=7.5)
    ax.set_xlabel("clipped area (%)", fontsize=7)
axes[0].set_ylabel("measure score", fontsize=7.5)
fig.suptitle("Clipping-dose response: 480 enhancement levels, 60 images",
             fontsize=8.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.86), pad=0.4, w_pad=0.8)
fig.savefig(os.path.join(HERE, "fig_dose_response.pdf"),
            bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---------------------------------------------------------------- koniq
with open(os.path.join(RES, "koniq_eaqi_results.json"), encoding="utf-8") as f:
    kq = json.load(f)
with open(os.path.join(RES, "koniq_baselines_seed0_test.json"), encoding="utf-8") as f:
    kb = json.load(f)

methods = ["NIQE", "BRISQUE", "PIQE", "PCoE (clip)", "E-AQI (trained)",
           "TOPIQ-NR"]
srcc = [kb["results"]["niqe"]["srcc"], kb["results"]["brisque"]["srcc"],
        kb["results"]["piqe"]["srcc"], kb["results"]["pcoe_clip"]["srcc"],
        kq["srcc"]["mean"], kb["results"]["topiq_nr"]["srcc"]]
err = [0, 0, 0, 0, kq["srcc"]["sd"], 0]

fig, ax = plt.subplots(figsize=(3.35, 2.0))
colors = ["#9ca3af", "#9ca3af", "#9ca3af", "#9ca3af", "#111827", "#6b7280"]
bars = ax.barh(range(len(methods)), srcc, xerr=err, color=colors,
               height=0.62, error_kw=dict(lw=1.0, capsize=2.5))
for i, v in enumerate(srcc):
    ax.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=6.8)
ax.axvline(0, color="#111827", lw=0.7)
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods)
ax.invert_yaxis()
ax.set_xlabel("SRCC on the held-out KonIQ-10k test partition ($n=2015$)",
              fontsize=7.4)
ax.set_xlim(-0.30, 1.06)
ax.set_title("Quality transfer on KonIQ-10k", loc="left", fontsize=8.2)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(HERE, "fig_koniq_srcc.pdf"),
            bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

print("figures written:",
      "fig_location_forest.pdf,", "fig_dose_response.pdf,",
      "fig_koniq_srcc.pdf")
