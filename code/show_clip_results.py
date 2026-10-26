# -*- coding: utf-8 -*-
"""Print the aggregate results of the clipping-location study."""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(ROOT, "results", "clipping_location_study.json")
d = json.load(open(path, encoding="utf-8"))

print("usable images:", d["protocol"]["n_base_images_usable"],
      "| runtime:", d["protocol"]["runtime_s"], "s")
print("-" * 70)
for name, agg in d["aggregate"].items():
    print(f"{name}: n={agg['n_images']}  "
          f"mean(T-E)={agg['mean_diff']:+.4f}  sd={agg['sd_diff']:.4f}  "
          f"Wilcoxon p={agg['wilcoxon_p']:.2e}  HL={agg['hodges_lehmann']:+.4f}  "
          f"CI95=[{agg['boot_ci95'][0]:+.4f}, {agg['boot_ci95'][1]:+.4f}]")
print("-" * 70)
# fraction of images where interior clipping scores worse (severity-up)
for name in d["aggregate"]:
    diffs = []
    for r in d["per_image"]:
        if name in r["metrics"]:
            s = r["metrics"][name]["scores"]
            diffs.append(s["T"] - s["E"])
    n_pos = sum(x > 0 for x in diffs)
    print(f"{name}: T worse than E in {n_pos}/{len(diffs)} images "
          f"({100 * n_pos / len(diffs):.1f}%)")
