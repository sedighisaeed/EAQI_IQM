# -*- coding: utf-8 -*-
"""Generate blank manual-entry templates for the E-AQI human study.

Creates two CSV files that mirror the platform's official export format
(GET /admin/export.csv) so hand-collected ratings can be typed in offline
and later imported into study.db with import_manual.py:

  manual_data_participants.csv  one row per observer (demographics)
  manual_data_ratings.csv       one row per rating (pre-filled with every
                                valid stimulus: 126 bases x 5 conditions
                                for participant P001, rating column blank)

Re-running this script OVERWRITES both templates (use --force only when
they are still empty; it refuses otherwise to protect entered data).
"""
from __future__ import annotations

import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "stimuli", "manifest.json")
RATINGS_CSV = os.path.join(HERE, "manual_data_ratings.csv")
PARTS_CSV = os.path.join(HERE, "manual_data_participants.csv")

CONDS = ["E", "T", "X", "N", "C"]

RATING_COLS = ["participant", "base", "cond", "is_repeat", "rating",
               "severity_delta", "rt_ms", "shown_left", "trial_index", "ts"]
PART_COLS = ["participant", "age", "gender", "vision_normal", "expertise",
             "acuity_ok", "display", "device", "finished", "withdrawn"]

# three example rows (participant P000) that import_manual.py skips by default
EXAMPLES = [
    ["P000", "23080", "E", "0", "42", "", "3500", "", "", ""],
    ["P000", "23080", "T", "0", "67", "", "4100", "", "", ""],
    ["P000", "23080", "ER", "1", "45", "", "3900", "", "", ""],  # repeat row
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite templates even if manual_data_ratings.csv "
                         "already exists")
    args = ap.parse_args()

    if os.path.exists(RATINGS_CSV) and not args.force:
        raise SystemExit(f"{RATINGS_CSV} already exists; pass --force to "
                         "overwrite (this discards any typed-in data)")

    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    bases = [s["base"] for s in man["stimuli"]]

    with open(RATINGS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(RATING_COLS)
        for ex in EXAMPLES:
            w.writerow(ex)
        for b in bases:                      # skeleton: every valid stimulus
            for c in CONDS:
                w.writerow(["P001", b, c, "0", "", "", "", "", "", ""])

    with open(PARTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(PART_COLS)
        w.writerow(["P000", "34", "female", "1", "none", "1",
                    "27-inch LCD sRGB", "desktop", "1", "0"])   # example
        w.writerow(["P001", "", "", "", "", "", "", "", "0", "0"])

    n = len(bases) * len(CONDS)
    print(f"wrote {os.path.basename(RATINGS_CSV)} "
          f"({len(EXAMPLES)} example rows + {n} blank rows for P001)")
    print(f"wrote {os.path.basename(PARTS_CSV)}")
    print("next: fill the rating column, then -> python import_manual.py")


if __name__ == "__main__":
    main()
