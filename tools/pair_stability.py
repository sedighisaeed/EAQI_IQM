# -*- coding: utf-8 -*-
"""Pair-level stability of the human T-E effect (reviewer request:
significance vs. practical importance). Computes, per base image, the mean
across observers of (T - E) and summarizes consistency across the 126 pairs.
"""
import io
import json
import os
import sqlite3
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "study_platform", "study.db")

con = sqlite3.connect(DB)
rows = con.execute(
    "SELECT participant, base, cond, rating FROM ratings "
    "WHERE is_repeat = 0 AND rating IS NOT NULL "
    "AND participant != 'P000'").fetchall()
con.close()

per = defaultdict(lambda: defaultdict(dict))  # p -> base -> cond -> rating
for p, b, c, r in rows:
    per[p][b][c] = float(r)

pair_diffs = {}  # base -> mean over observers of (T-E) for that base
for p, by_base in per.items():
    for b, conds in by_base.items():
        if "T" in conds and "E" in conds:
            pair_diffs.setdefault(b, []).append(conds["T"] - conds["E"])

bases = sorted(pair_diffs)
means = np.array([np.mean(pair_diffs[b]) for b in bases])
out = {
    "n_pairs": len(bases),
    "median_observers_per_pair": float(np.median([len(pair_diffs[b]) for b in bases])),
    "pair_mean_T_minus_E": {
        "median": float(np.median(means)),
        "iqr": [float(np.percentile(means, 25)), float(np.percentile(means, 75))],
        "min": float(means.min()), "max": float(means.max()),
        "pct_pairs_positive": float(100.0 * np.mean(means > 0)),
    },
    "note": ("pair-level effect = mean over observers of (T - E) per base "
             "image; stability summary over the 126 matched pairs"),
}
with io.open(os.path.join(HERE, "..", "results",
                          "human_pair_stability.json"), "w",
             encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
