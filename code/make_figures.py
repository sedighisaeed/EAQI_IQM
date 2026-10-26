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
ROOT = os.path.dirname(HERE)
RES = os.path.normpath(os.path.join(ROOT, "results"))
OUT = os.path.normpath(os.path.join(ROOT, "manuscript", "latex"))

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
# merge the executed E-AQI / PCoE rows from eval_eaqi_studies.py so that the
# forest plot shows all six measures
with open(os.path.join(RES, "eaqi_studies12.json"), encoding="utf-8") as f:
    st12 = json.load(f)
agg2 = st12["study1_location"]["aggregate"]
agg.update({
    "pcoe": agg2["pcoe"],
    "eaqi": agg2["eaqi"],
})
order = ["brisque", "niqe", "piqe", "pcoe", "topiq_nr", "eaqi"]
labels = {"brisque": "BRISQUE", "niqe": "NIQE", "piqe": "PIQE",
          "pcoe": "PCoE", "topiq_nr": "TOPIQ-NR", "eaqi": "E-AQI (trained)"}
FLIP = {"topiq_nr", "eaqi"}   # higher-is-better measures: plot on severity sign
# For lower-better measures, more severe = higher raw score, so T-E on the raw
# scale already follows the "severity" sign. TOPIQ-NR is higher-better, so its
# severity-scale difference is -(T-E). We plot all measures on one severity
# axis (positive = interior-texture clipping rated more severe).
fig, ax = plt.subplots(figsize=(3.35, 1.9))
for i, m in enumerate(order):
    a = agg[m]
    mean = a["mean_diff"]
    lo, hi = a["boot_ci95"]
    if m in FLIP:
        mean, lo, hi = -mean, -hi, -lo
    p = a["wilcoxon_p"]
    sig = p < 0.05
    ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]],
                fmt="o", ms=4.5, lw=1.4, capsize=2.5,
                color="#1f2937" if sig else "#9ca3af",
                mfc="#1f2937" if sig else "white", zorder=3)
    ax.annotate(f"$p={p:.2g}$", (hi, i), xytext=(4, -2),
                textcoords="offset points", fontsize=6.5,
                color="#1f2937" if sig else "#6b7280")
ax.axvline(0, color="#111827", lw=0.7, ls="--", zorder=1)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([labels[m] for m in order])
ax.invert_yaxis()
ax.set_xlabel("Paired effect on the severity scale: interior texture $-$ edge"
              "\n(native units; TOPIQ-NR sign-flipped), 95\\% bootstrap CI"
              .replace("\\%", "%"))
fig.set_figheight(2.6)
ax.set_title("Clipping-location effect (168 area-matched pairs)", loc="left")
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, "fig_location_forest.pdf"),
            bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# ---------------------------------------------------------------- study 2
with open(os.path.join(RES, "enhancement_clip_response.json"), encoding="utf-8") as f:
    dose = json.load(f)
rows = dose["rows"]
# 2 rows x 3 cols grid so each panel stays readable at \textwidth
dose_order = ["brisque", "niqe", "piqe", "pcoe", "topiq_nr", "eaqi"]
dose_labels = {"brisque": "BRISQUE", "niqe": "NIQE", "piqe": "PIQE",
              "pcoe": "PCoE", "topiq_nr": "TOPIQ-NR", "eaqi": "E-AQI"}
# merge the executed E-AQI / PCoE per-observation scores and rank correlations
s12dose = st12["study2_dose"]["srcc_clipfrac_vs_score"]
extras = st12["study2_dose"]["rows"]
dose_rows = []
for r, ex in zip(rows, extras):
    sc = dict(r["scores"])
    sc["pcoe"] = ex["pcoe"]
    sc["eaqi"] = ex["eaqi"]
    dose_rows.append({"clip_frac": r["clip_frac"], "scores": sc})
fig, axes = plt.subplots(2, 3, figsize=(7.05, 3.6), sharey=False)
for ax, m in zip(axes.ravel(), dose_order):
    cf = [r["clip_frac"] * 100 for r in dose_rows]
    sc = [r["scores"][m] for r in dose_rows]
    ax.scatter(cf, sc, s=4, alpha=0.45, color="#374151", linewidths=0)
    if m in s12dose:
        d = s12dose[m]
        rho, p = d["srcc_clipfrac_vs_score"], d["p_value"]
    else:
        d = [s for s in dose["srcc_clipfrac_vs_score"]
             if s["metric"] == m][0]
        rho, p = d["srcc_clipfrac_vs_score"], d["p_value"]
    ax.set_title(f"{dose_labels[m]}: $\\rho={rho:+.3f}$, $p={p:.1g}$",
                 fontsize=7.5, pad=2.5)
    ax.set_xlabel("clipped area (%)", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6.5)
for ax in axes[:, 0]:
    ax.set_ylabel("measure score", fontsize=7.5, labelpad=1)
fig.suptitle("Clipping-dose response: 480 enhancement levels, 60 images",
             fontsize=8.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.4, w_pad=1.0, h_pad=1.0)
fig.savefig(os.path.join(OUT, "fig_dose_response.pdf"),
            bbox_inches="tight", pad_inches=0.03)
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
ax.set_xlabel("SRCC on the held-out KonIQ-10k test partition\n($n=2{,}015$)")
ax.set_xlim(-0.28, 1.0)
ax.set_title("Quality transfer on KonIQ-10k", loc="left")
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, "fig_koniq_srcc.pdf"),
            bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

print("figures written:",
      "fig_location_forest.pdf,", "fig_dose_response.pdf,",
      "fig_koniq_srcc.pdf")
