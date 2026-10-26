# -*- coding: utf-8 -*-
"""Merge seed-0 (backup) with the freshly trained seeds 1/2 into the final
three-seed koniq_eaqi_results.json, and merge the seed-0 ablation reference
row into the fresh ablation JSON if needed.
"""
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))

# ---- koniq three-seed merge ----
with io.open(os.path.join(RES, "koniq_eaqi_results_seed0_backup.json"),
             encoding="utf-8") as f:
    seed0 = json.load(f)
with io.open(os.path.join(RES, "koniq_eaqi_results.json"),
             encoding="utf-8") as f:
    new = json.load(f)

merged = dict(new)
merged["per_seed"] = {}
merged["per_seed"].update(seed0.get("per_seed", {}))
merged["per_seed"].update(new.get("per_seed", {}))
seeds = sorted(merged["per_seed"], key=int)
merged["n_seeds"] = len(seeds)
merged["args"]["seeds"] = [int(s) for s in seeds]
merged["args"]["note"] = ("seeds 0 from the original run and seeds 1-2 from "
                          "the completion run share the identical split and "
                          "protocol")
for k in ("srcc", "krcc", "plcc", "rmse"):
    vals = np.array([merged["per_seed"][s][k] for s in seeds])
    merged[k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1))}

with io.open(os.path.join(RES, "koniq_eaqi_results.json"), "w",
             encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print("merged seeds:", seeds)
for k in ("srcc", "krcc", "plcc", "rmse"):
    print(f"  {k}: {merged[k]['mean']:.4f} +- {merged[k]['sd']:.4f}")
for s in seeds:
    r = merged["per_seed"][s]
    print(f"  seed {s}: srcc={r['srcc']:.4f} krcc={r['krcc']:.4f} "
          f"plcc={r['plcc']:.4f} rmse={r['rmse']:.4f}")
