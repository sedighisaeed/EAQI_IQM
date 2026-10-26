# -*- coding: utf-8 -*-
"""Summarize all experimental results JSONs for the manuscript tables."""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


print("=" * 72)
print("1) ENHANCEMENT CLIP RESPONSE (gamma sweep)")
print("=" * 72)
d = load("enhancement_clip_response.json")
if d:
    for r in d["srcc_clipfrac_vs_score"]:
        print(f"  {r['metric']}: SRCC(clipfrac, score) = "
              f"{r['srcc_clipfrac_vs_score']:+.4f}  (p={r['p_value']:.2e}, "
              f"n={r['n_obs']})")
    # mean score per gamma level
    gammas = sorted({round(r["gamma"], 4) for r in d["rows"]})
    print("  per-level means:")
    for g in gammas:
        rr = [r for r in d["rows"] if abs(r["gamma"] - g) < 1e-6]
        if not rr:
            continue
        line = f"    gamma={g:.3f} clipfrac={sum(x['clip_frac'] for x in rr) / len(rr):.4f}: "
        for m in ("brisque", "niqe", "piqe", "topiq_nr"):
            vals = [x["scores"][m] for x in rr if m in x["scores"]]
            if vals:
                line += f"{m}={sum(vals) / len(vals):+.3f} "
        print(line)

print()
print("=" * 72)
print("2) GPU PROFILE")
print("=" * 72)
d = load("latency_gpu.json")
if d:
    print(" ", json.dumps(d, indent=2))
