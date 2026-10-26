# -*- coding: utf-8 -*-
"""Import hand-typed manual_data CSVs into the platform's study.db.

Reads the two templates produced by make_manual_template.py
(manual_data_ratings.csv, manual_data_participants.csv), validates every
row against the stimulus manifest and the pre-registered rating range,
then writes rows in the exact same schema the live server uses, so
analysis.py and the admin dashboard work unchanged.

Validation rules (a bad row aborts the import; nothing is committed):
  * participant id must be listed in the participants CSV
  * base must exist in stimuli/manifest.json, cond in {E,T,X,N,C}
  * repeats: cond must be E_R or T_R with is_repeat=1
  * rating must be an integer 0-100 (2AFC rows: -1..1 with shown_left)
  * trial_index is optional; if blank it is auto-assigned per participant

Usage:
  python import_manual.py                 # validate + import
  python import_manual.py --dry-run       # validate only, no DB writes
  python import_manual.py --ratings x.csv --participants y.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "study.db")
MANIFEST = os.path.join(HERE, "stimuli", "manifest.json")
DEFAULT_RATINGS = os.path.join(HERE, "manual_data_ratings.csv")
DEFAULT_PARTS = os.path.join(HERE, "manual_data_participants.csv")

CONDS = {"E", "T", "X", "N", "C"}
REPEAT_TAG = "R"              # same tag the live server appends for repeats
EXAMPLE_PID = "P000"          # example rows in the template; never imported
REQUIRED_PART_COLS = ["participant"]
REQUIRED_RATING_COLS = ["participant", "base", "cond", "rating"]


class ImportError_(SystemExit):
    pass


def load_manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def parse_int(value, field: str, row_no: int, lo: int, hi: int):
    v = (value or "").strip()
    if v == "":
        return None
    try:
        n = int(float(v))
    except ValueError:
        raise ImportError_(f"row {row_no}: {field}={value!r} is not an integer")
    if not (lo <= n <= hi):
        raise ImportError_(f"row {row_no}: {field}={n} outside [{lo}, {hi}]")
    return n


def read_csv(path: str, required: list[str]) -> list[dict]:
    if not os.path.exists(path):
        raise ImportError_(f"missing file: {path}")
    with open(path, encoding="utf-8-sig") as f:      # tolerate Excel BOM
        rows = list(csv.DictReader(f))
    if not rows:
        raise ImportError_(f"{path}: no data rows")
    header = list(rows[0].keys())
    missing = [c for c in required if c not in header]
    if missing:
        raise ImportError_(f"{path}: missing column(s) {missing}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=DEFAULT_RATINGS)
    ap.add_argument("--participants", default=DEFAULT_PARTS)
    ap.add_argument("--db", default=DB_PATH,
                    help="target SQLite database (default: study.db)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate only; do not write to the database")
    args = ap.parse_args()

    man = load_manifest()
    valid_bases = {s["base"] for s in man["stimuli"]}

    # ---- participants ----
    part_rows = read_csv(args.participants, REQUIRED_PART_COLS)
    pids = []
    for i, r in enumerate(part_rows, 2):
        pid = (r.get("participant") or "").strip()
        if not pid:
            raise ImportError_(f"participants row {i}: empty participant id")
        pids.append(pid)
    known_pids = {p for p in pids if p != EXAMPLE_PID}
    if not known_pids:
        raise ImportError_("no real participants found (only the P000 "
                           "example row); add your observers first")
    if len(known_pids) != len(pids) - pids.count(EXAMPLE_PID):
        raise ImportError_("duplicate participant ids in participants CSV")

    # ---- ratings ----
    rating_rows = read_csv(args.ratings, REQUIRED_RATING_COLS)
    prepared: list[tuple] = []
    n_examples = 0
    n_unfilled = 0
    per_p_trial = {p: 0 for p in known_pids}
    seen = set()
    for i, r in enumerate(rating_rows, 2):
        pid = (r.get("participant") or "").strip()
        base = (r.get("base") or "").strip()
        cond = (r.get("cond") or "").strip().upper()
        raw_rating = (r.get("rating") or "").strip()
        if not any([pid, base, cond, raw_rating]):
            continue                                  # fully blank row
        if pid == EXAMPLE_PID:
            n_examples += 1
            continue                                  # example row: skipped
        if pid not in known_pids:
            raise ImportError_(f"ratings row {i}: participant {pid!r} is not "
                               "in the participants CSV")
        if base not in valid_bases:
            raise ImportError_(f"ratings row {i}: base {base!r} not in "
                               "stimuli/manifest.json")
        is_repeat = 0
        stem = cond[:-2] if cond.endswith("_R") else cond
        if stem != cond and stem in CONDS:
            is_repeat, stem = 1, stem          # explicit E_R / T_R form
        elif len(cond) == 2 and cond.endswith(REPEAT_TAG) and cond[0] in CONDS:
            is_repeat, stem = 1, cond[0]       # platform-canonical ER / TR form
        if stem not in CONDS:
            raise ImportError_(f"ratings row {i}: cond {cond!r} invalid "
                               "(use E/T/X/N/C, or ER/TR for repeats)")
        if is_repeat and stem not in ("E", "T"):
            raise ImportError_(f"ratings row {i}: only E/T repeats are used "
                               "by the platform")
        db_cond = stem + (REPEAT_TAG if is_repeat else "")   # canonical form
        if raw_rating == "":
            n_unfilled += 1
            continue                          # skeleton row: not yet filled in
        rating = parse_int(raw_rating, "rating", i, 0, 100)
        sev = parse_int(r.get("severity_delta"), "severity_delta", i, 0, 100)
        rt = parse_int(r.get("rt_ms"), "rt_ms", i, 0, 10 ** 9)
        shown_left = (r.get("shown_left") or "").strip() or None
        ti = parse_int(r.get("trial_index"), "trial_index", i, 0, 10 ** 6)
        if ti is None:
            ti = per_p_trial[pid]
            per_p_trial[pid] += 1
        key = (pid, base, db_cond, is_repeat)
        if key in seen:
            raise ImportError_(f"ratings row {i}: duplicate trial "
                               f"{pid}/{base}/{cond}")
        seen.add(key)
        prepared.append((pid, base, db_cond, is_repeat, rating, sev, rt,
                         shown_left, ti))

    print(f"participants : {len(known_pids)} (ids: {', '.join(sorted(known_pids))})")
    print(f"rating rows  : {len(prepared)} valid "
          f"({n_examples} example rows skipped, "
          f"{n_unfilled} skeleton rows still unfilled)")
    per_p = {}
    for pid, *_ in prepared:
        per_p[pid] = per_p.get(pid, 0) + 1
    for pid in sorted(per_p):
        print(f"  {pid}: {per_p[pid]} ratings")
    if args.dry_run:
        print("dry-run OK — no data written")
        return

    conn = sqlite3.connect(args.db)
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS participants (
                  id TEXT PRIMARY KEY, created_at REAL, consent INTEGER,
                  demographics TEXT, acuity_ok INTEGER, display TEXT,
                  finished INTEGER DEFAULT 0, withdrawn INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS ratings (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  participant TEXT, base TEXT, cond TEXT, is_repeat INTEGER,
                  rating INTEGER, severity_delta INTEGER, rt_ms INTEGER,
                  shown_left TEXT, trial_index INTEGER, ts REAL);
                CREATE INDEX IF NOT EXISTS idx_ratings_p
                  ON ratings(participant);""")
            now = time.time()
            for r in part_rows:
                pid = (r.get("participant") or "").strip()
                if pid == EXAMPLE_PID or pid not in known_pids:
                    continue
                demo = json.dumps({k: (r.get(k) or "").strip()
                                   for k in ("age", "gender", "vision_normal",
                                             "expertise", "device")},
                                  ensure_ascii=False)
                conn.execute(
                    "INSERT OR IGNORE INTO participants(id, created_at,"
                    " consent, demographics, acuity_ok, display, finished)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (pid, now, 1, demo,
                     1 if (r.get("acuity_ok") or "").strip() in ("1", "true", "yes") else 0,
                     (r.get("display") or "").strip(),
                     1 if (r.get("finished") or "").strip() in ("1", "true", "yes") else 0))
            conn.executemany(
                "INSERT INTO ratings(participant, base, cond, is_repeat,"
                " rating, severity_delta, rt_ms, shown_left, trial_index, ts)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(*row, now) for row in prepared])
    finally:
        conn.close()
    print(f"imported -> {os.path.normpath(args.db)}")
    print("next: python analysis.py --db study.db --out ../results/manual_analysis")


if __name__ == "__main__":
    main()
