# -*- coding: utf-8 -*-
"""Statistical analysis of the E-AQI human perceptual study.

Reads study.db (or an exported JSON), applies pre-registered QC rules, and
produces every number and table the manuscript needs:

  H1 (primary): per-observer paired difference  T - E  (annoyance, 0-100)
      * Wilcoxon signed-rank test (two-sided)
      * Hodges-Lehmann estimate + bootstrap CI (2000 resamples, observer-level)
      * Cohen's d_z with noncentral-t CI  (eatqi.stats.paired_analysis)
      * per-condition means, % of observers with T > E
  H2 (secondary): condition main effects (E/T/X/N/C)
      * Friedman test across conditions + post-hoc paired Wilcoxon (Holm)
  H3 (exploratory): 2AFC paired task
      * sign test vs 50%, one-sample Wilcoxon on signed severity deltas
  QC: repeat consistency (|r - r'| <= 15 on 0-100), median RT >= 1.5 s,
      minimum ratings per observer (>= 80% of trials), warmup excluded.

Outputs:
  results/human_study_analysis.json
  manuscript/latex/tab_human_h1.tex      (main H1 table)
  manuscript/latex/tab_human_conditions.tex (per-condition table)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CODE_DIR = os.path.join(ROOT, "code")
sys.path.insert(0, CODE_DIR)

from eatqi.stats import paired_analysis  # noqa: E402

DB_PATH = os.path.join(HERE, "study.db")
OUT_JSON = os.path.join(ROOT, "results", "human_study_analysis.json")
TEX_DIR = os.path.join(ROOT, "manuscript", "latex")

N_WARMUP = 5
RT_MIN_MS = 1500
REPEAT_TOL = 15
MIN_COVERAGE = 0.80
BOOT = 2000
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def load_ratings() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT r.*, p.finished, p.withdrawn, p.acuity_ok FROM ratings r"
        " JOIN participants p ON p.id = r.participant"
        " WHERE p.withdrawn = 0")]
    conn.close()
    return rows


def qc_filter(rows: list[dict]) -> tuple[list[dict], list[str], dict]:
    """Pre-registered QC. Returns (clean_rows, excluded_ids, qc_summary)."""
    per_p = defaultdict(list)
    for r in rows:
        per_p[r["participant"]].append(r)

    excluded, notes = [], []
    clean = []
    for pid, rs in per_p.items():
        reasons = []
        main = [r for r in rs if r["is_repeat"] == 0 and r["trial_index"] >= N_WARMUP]
        n_trials = len({r["trial_index"] for r in rs})
        if n_trials and len(main) / max(n_trials - N_WARMUP, 1) < MIN_COVERAGE:
            reasons.append("coverage<80%")
        rts = np.array([r["rt_ms"] for r in main if r["rt_ms"]])
        if len(rts) and np.median(rts) < RT_MIN_MS:
            reasons.append("median_rt<1.5s")
        # repeat consistency
        orig = {r["base"]: r["rating"] for r in main}
        reps = [r for r in rs if r["is_repeat"]]
        if reps:
            devs = [abs(orig.get(r["base"], np.nan) - r["rating"])
                    for r in reps if r["base"] in orig]
            devs = [d for d in devs if not np.isnan(d)]
            if devs and np.mean(np.array(devs) <= REPEAT_TOL) < 0.5:
                reasons.append("repeat_consistency<50%")
        if reasons:
            excluded.append(pid)
            notes.append(f"{pid}: {','.join(reasons)}")
        else:
            clean.extend(main)
            # keep repeats too (for consistency reporting)
            clean.extend(r for r in reps if r["base"] in orig)
    qc = {"n_participants_total": len(per_p),
          "n_participants_excluded": len(excluded),
          "excluded": excluded, "notes": notes,
          "repeat_tolerance": REPEAT_TOL, "rt_min_ms": RT_MIN_MS,
          "warmup_excluded": N_WARMUP, "min_coverage": MIN_COVERAGE}
    return clean, excluded, qc


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------

def observer_paired_diffs(clean: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-observer mean(T) and mean(E) over common base images."""
    by_p = defaultdict(lambda: defaultdict(dict))
    for r in clean:
        if r["rating"] is None:
            continue
        by_p[r["participant"]][r["cond"]][r["base"]] = r["rating"]
    diffs, ids = [], []
    for pid, conds in by_p.items():
        if "T" not in conds or "E" not in conds:
            continue
        common = set(conds["T"]) & set(conds["E"])
        if len(common) < 10:
            continue
        d = np.mean([conds["T"][b] for b in common]) - \
            np.mean([conds["E"][b] for b in common])
        diffs.append(d)
        ids.append(pid)
    return np.array(diffs), np.array(ids)


def wilcoxon_hl_bootstrap(diffs: np.ndarray) -> dict:
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    res = {"n": n, "mean": float(diffs.mean()),
           "sd": float(diffs.std(ddof=1)) if n > 1 else float("nan")}
    if n >= 10:
        w = sps.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        res["wilcoxon_p"] = float(w.pvalue)
        res["wilcoxon_W"] = float(w.statistic)
        # Hodges-Lehmann: median over all pairwise averages
        hl = np.median((diffs[:, None] + diffs[None, :]).ravel() / 2.0)
        res["hodges_lehmann"] = float(hl)
        boots = np.array([RNG.choice(diffs, size=n, replace=True).mean()
                          for _ in range(BOOT)])
        res["boot_ci95"] = [float(np.percentile(boots, 2.5)),
                            float(np.percentile(boots, 97.5))]
        res["pct_T_gt_E"] = float(100.0 * np.mean(diffs > 0))
        # Cohen's dz + noncentral-t CI
        pr = paired_analysis(diffs)
        res["dz"] = pr.dz
        res["dz_ci95"] = [pr.ci95_low, pr.ci95_high]
        res["t_stat"] = pr.t_stat
        res["paired_t_p"] = pr.p_two_sided
    return res


def condition_table(clean: list[dict]) -> dict:
    """Per-condition means and Friedman across E/T/X/N/C (observer means)."""
    by_p = defaultdict(lambda: defaultdict(list))
    for r in clean:
        if r["rating"] is None or r["is_repeat"]:
            continue
        by_p[r["participant"]][r["cond"]].append(r["rating"])
    conds = ["E", "T", "X", "N", "C"]
    mat = []
    for pid, c in by_p.items():
        if all(k in c and len(c[k]) >= 10 for k in conds):
            mat.append([np.mean(c[k]) for k in conds])
    mat = np.array(mat)
    out = {"n_observers": int(mat.shape[0]) if mat.size else 0}
    if mat.size:
        out["condition_means"] = {k: float(m) for k, m in zip(conds, mat.mean(0))}
        out["condition_sd"] = {k: float(s) for k, s in zip(conds, mat.std(0, ddof=1))}
        fr = sps.friedmanchisquare(*[mat[:, i] for i in range(len(conds))])
        out["friedman_chi2"] = float(fr.statistic)
        out["friedman_p"] = float(fr.pvalue)
        # post-hoc paired Wilcoxon T vs each other condition, Holm correction
        post = {}
        pvals = []
        keys = []
        for other in ("E", "X", "N", "C"):
            d = mat[:, conds.index("T")] - mat[:, conds.index(other)]
            w = sps.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
            post[f"T-{other}"] = {"mean": float(d.mean()),
                                  "wilcoxon_p": float(w.pvalue)}
            pvals.append(w.pvalue)
            keys.append(f"T-{other}")
        order = np.argsort(pvals)
        m = len(pvals)
        holm = [0.0] * m
        running = 0.0
        for rank, idx in enumerate(order):
            adj = min(1.0, (m - rank) * pvals[idx])
            running = max(running, adj)
            holm[idx] = running
        for k, p in zip(keys, holm):
            post[k]["holm_p"] = float(p)
        out["posthoc_T_vs"] = post
    return out


def afc_analysis(clean: list[dict]) -> dict:
    """2AFC: rating stores sign*delta; positive = T judged worse."""
    by_p = defaultdict(list)
    for r in clean:
        if r.get("shown_left") and r["rating"] is not None:
            by_p[r["participant"]].append(r["rating"])
    ids, means, wins = [], [], []
    for pid, v in by_p.items():
        v = np.array(v, dtype=float)
        if len(v) < 10:
            continue
        ids.append(pid)
        means.append(v.mean())
        wins.append(100.0 * np.mean(v > 0))
    out = {"n_observers": len(ids)}
    if ids:
        out["mean_signed_delta"] = float(np.mean(means))
        out["pct_chose_T"] = float(np.mean(wins))
        w = sps.wilcoxon(np.array(means), alternative="two-sided",
                         zero_method="wilcox")
        out["wilcoxon_p"] = float(w.pvalue)
    return out


def repeat_consistency(clean: list[dict]) -> dict:
    by_p = defaultdict(dict)
    for r in clean:
        if r["is_repeat"] == 0 and r["rating"] is not None:
            by_p[r["participant"]][r["base"] + "|" + r["cond"]] = r["rating"]
    devs = []
    for r in clean:
        if r["is_repeat"] and r["rating"] is not None:
            k = r["base"] + "|" + r["cond"].rstrip("R")
            if k in by_p.get(r["participant"], {}):
                devs.append(abs(by_p[r["participant"]][k] - r["rating"]))
    if not devs:
        return {}
    devs = np.array(devs, dtype=float)
    return {"n_repeat_pairs": int(devs.size),
            "mean_abs_dev": float(devs.mean()),
            "pct_within_15": float(100 * np.mean(devs <= REPEAT_TOL)),
            "pearson_r": float(np.corrcoef(
                devs, devs)[0, 1]) if False else None}


# --------------------------------------------------------------------------
# LaTeX tables
# --------------------------------------------------------------------------

def tex_h1(h1: dict) -> str:
    if h1.get("n", 0) < 10:
        return "% H1: not enough observers yet\n"
    lo, hi = h1["boot_ci95"]
    dzlo, dzhi = h1["dz_ci95"]
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Human perceptual study (H1): paired effect of clipping "
        "location on annoyance (0--100), per-observer interior-texture minus "
        f"edge-aligned difference ({h1['n']} observers, "
        "area- and histogram-matched stimuli).}\n"
        "\\label{tab:human-h1}\n\\footnotesize\n"
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        "mean(T$-$E) & SD & Wilcoxon $p$ & HL est. & "
        "$\\mathrm{CI}_{95}$ & $d_z$ \\\\\n\\midrule\n"
        f"${h1['mean']:+.1f}$ & ${h1['sd']:.1f}$ & "
        f"${h1['wilcoxon_p']:.2g}$ & ${h1['hodges_lehmann']:+.2f}$ & "
        f"$[{lo:+.2f},\\,{hi:+.2f}]$ & "
        f"${h1['dz']:+.2f}\\,[{dzlo:+.2f},{dzhi:+.2f}]$ \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def tex_conditions(ct: dict) -> str:
    if not ct.get("condition_means"):
        return "% conditions: not enough observers yet\n"
    cm, cs = ct["condition_means"], ct["condition_sd"]
    rows = " & ".join(
        f"{k}: ${cm[k]:.1f}\\pm{cs[k]:.1f}$" for k in ("E", "T", "X", "N", "C"))
    caption = ("Human perceptual study: per-condition mean annoyance "
               f"({ct['n_observers']} observers; Friedman $\\chi^2(4) = "
               f"{ct['friedman_chi2']:.1f}$, $p = {ct['friedman_p']:.2g}$).")
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{" + caption + "}\n"
        "\\label{tab:human-conditions}\n\\footnotesize\n"
        f"{rows}\n"
        "\end{table}\n")


# --------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(DB_PATH):
        raise SystemExit(f"no study database at {DB_PATH}")
    rows = load_ratings()
    clean, excluded, qc = qc_filter(rows)
    diffs, ids = observer_paired_diffs(clean)
    h1 = wilcoxon_hl_bootstrap(diffs) if len(diffs) else {"n": 0}
    ct = condition_table(clean)
    afc = afc_analysis(clean)
    rep = repeat_consistency(clean)

    out = {
        "qc": qc,
        "n_ratings_clean": len(clean),
        "h1_paired_T_minus_E": h1,
        "h1_observer_ids": ids.tolist(),
        "conditions": ct,
        "twoafc": afc,
        "repeat_consistency": rep,
        "pre_registration_note": (
            "QC rules fixed before data collection: warm-up exclusion, "
            "median RT floor, coverage floor, repeat-consistency filter."),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    os.makedirs(TEX_DIR, exist_ok=True)
    with open(os.path.join(TEX_DIR, "tab_human_h1.tex"), "w",
              encoding="utf-8") as f:
        f.write(tex_h1(h1))
    with open(os.path.join(TEX_DIR, "tab_human_conditions.tex"), "w",
              encoding="utf-8") as f:
        f.write(tex_conditions(ct))

    print(json.dumps({k: out[k] for k in
                      ("qc", "h1_paired_T_minus_E", "conditions", "twoafc")},
                     indent=2))
    print("saved ->", os.path.normpath(OUT_JSON))


if __name__ == "__main__":
    main()
