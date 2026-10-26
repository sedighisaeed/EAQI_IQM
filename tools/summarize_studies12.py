# -*- coding: utf-8 -*-
"""Summarize study2_dose section of eaqi_studies12.json to stdout."""
import io
import json
import os

import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "..", "results", "eaqi_studies12.json")

d = json.load(io.open(PATH, encoding="utf-8"))
s2 = d["study2_dose"]
out = {k: v for k, v in s2.items() if k != "per_image"}

# dose-response: SRCC of clipped fraction vs score, E-AQI and PCoE
rows = s2.get("per_image", [])
if rows:
    sample = rows[0]
    out["per_image_keys"] = list(sample.keys())
    if "gamma" in sample:
        gammas = sorted({r["gamma"] for r in rows})
        out["gammas"] = gammas
        eaqi_curve, pcoe_curve, clip_curve = [], [], []
        for g in gammas:
            gr = [r for r in rows if r["gamma"] == g]
            eaqi_curve.append(float(np.mean([r["scores"]["eaqi"] for r in gr])))
            pcoe_curve.append(float(np.mean([r["scores"]["pcoe"] for r in gr])))
            clip_curve.append(float(np.mean([r["clip_frac"] for r in gr])))
        out["mean_eaqi_by_gamma"] = eaqi_curve
        out["mean_pcoe_by_gamma"] = pcoe_curve
        out["mean_clipfrac_by_gamma"] = clip_curve
        # srcc clip fraction vs score pooled
        clip = np.array([r["clip_frac"] for r in rows])
        eaq = np.array([r["scores"]["eaqi"] for r in rows])
        pc = np.array([r["scores"]["pcoe"] for r in rows])
        out["srcc_clip_vs_eaqi"] = sps.spearmanr(clip, eaq)._asdict()
        out["srcc_clip_vs_pcoe"] = sps.spearmanr(clip, pc)._asdict()

print(json.dumps(out, indent=2))
