# -*- coding: utf-8 -*-
"""E-AQI human perceptual study — self-contained web platform.

Zero external dependencies: Python standard library only (http.server + sqlite3).

Routes (observer):
  GET  /                     observer landing: consent + acuity + demographics
  GET  /study                rating interface (absolute 0-100 slider)
  GET  /study2afc            paired 2AFC interface (optional secondary task)
  POST /api/register         create participant (consent recorded)
  GET  /api/next             next stimulus for this participant (balanced order)
  POST /api/rate             submit one rating (or one 2AFC response)
  GET  /api/progress         participant progress
  GET  /stimuli/<file>.jpg   stimulus bitmaps

Routes (admin, token-protected):
  GET  /admin                dashboard (progress per participant)
  GET  /admin/export.json    full raw export
  GET  /admin/export.csv     flat ratings CSV
  POST /admin/reset           delete one participant's data (withdrawal)

Run:
  python server.py --port 8000 --admin-token study2025
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import string
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "study.db")
STIM_DIR = os.path.join(HERE, "stimuli")
STATIC_DIR = os.path.join(HERE, "static")

N_WARMUP = 5
REPEAT_FRACTION = 0.20        # 20% intra-observer repeats
REPEAT_TAG = "R"              # condition tag for repeat presentations
BLOCK_SIZE = 240              # incomplete-block cap per observer (0 = full)

_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
  id          TEXT PRIMARY KEY,
  created_at  REAL,
  consent     INTEGER,
  demographics TEXT,
  acuity_ok   INTEGER,
  display     TEXT,
  finished    INTEGER DEFAULT 0,
  withdrawn   INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ratings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  participant TEXT,
  base TEXT,
  cond  TEXT,             -- E/T/X/N/C (or E_R/T_R for repeats)
  is_repeat INTEGER,
  rating INTEGER,         -- 0-100 annoyance (or -1..1 choice for 2AFC)
  severity_delta INTEGER, -- 2AFC only: perceived magnitude 0-100
  rt_ms   INTEGER,
  shown_left  TEXT,       -- 2AFC only
  trial_index INTEGER,
  ts REAL
);
CREATE INDEX IF NOT EXISTS idx_ratings_p ON ratings(participant);
"""


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _LOCK, db() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------
# stimulus playlist (balanced, per participant)
# --------------------------------------------------------------------------

def load_manifest() -> dict:
    with open(os.path.join(STIM_DIR, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def build_playlist(playlist_seed: str) -> list[dict]:
    """Deterministic per-participant shuffled playlist with repeats.

    With BLOCK_SIZE > 0 (default 240) a counterbalanced incomplete-block
    design is used: the shuffled full playlist is taken modulo a rotation
    determined by the participant seed, so successive participants cover the
    full stimulus set while each session stays near the 24-minute cap.
    BLOCK_SIZE = 0 serves the full single-session playlist (laboratory mode).
    """
    man = load_manifest()
    bases = [s["base"] for s in man["stimuli"]]
    rng = random.Random(playlist_seed)

    trials = []
    for base in bases:
        for cond in ("E", "T", "X", "N", "C"):
            trials.append({"base": base, "cond": cond, "is_repeat": 0})
    rng.shuffle(trials)

    if BLOCK_SIZE and len(trials) > BLOCK_SIZE:
        rot = rng.randrange(len(trials))
        trials = trials[rot:] + trials[:rot]
        trials = trials[:BLOCK_SIZE]

    # inject 20% repeats of the H1-relevant conditions (E/T) at random spots
    n_rep = int(len(trials) * REPEAT_FRACTION)
    rep_pool = [t for t in trials if t["cond"] in ("E", "T")]
    reps = []
    for t in rng.sample(rep_pool, min(n_rep, len(rep_pool))):
        reps.append({"base": t["base"], "cond": t["cond"] + REPEAT_TAG,
                     "is_repeat": 1})
    # spread repeats evenly through the second half
    for i, r in enumerate(reps):
        pos = len(trials) // 2 + int(i * (len(trials) // 2) / max(len(reps), 1))
        trials.insert(min(pos, len(trials)), r)
    return trials


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "EAQIStudy/1.0"

    # ---------- helpers ----------
    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: str, ctype: str) -> None:
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self._json({"error": "not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quiet
        pass

    # ---------- routing ----------
    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        if path == "/":
            self._file(os.path.join(STATIC_DIR, "landing.html"), "text/html; charset=utf-8")
        elif path == "/study":
            self._file(os.path.join(STATIC_DIR, "study.html"), "text/html; charset=utf-8")
        elif path == "/study2afc":
            self._file(os.path.join(STATIC_DIR, "study2afc.html"), "text/html; charset=utf-8")
        elif path == "/admin":
            self._file(os.path.join(STATIC_DIR, "admin.html"), "text/html; charset=utf-8")
        elif path == "/api/manifest":
            self._json(load_manifest())
        elif path == "/api/next":
            q = parse_qs(u.query)
            self._next(q.get("p", [""])[0])
        elif path == "/api/progress":
            q = parse_qs(u.query)
            self._progress(q.get("p", [""])[0])
        elif path == "/admin/export.json":
            if self._admin_ok(u):
                self._export_json()
        elif path == "/admin/export.csv":
            if self._admin_ok(u):
                self._export_csv()
        elif path.startswith("/stimuli/"):
            name = os.path.basename(path)
            if re.fullmatch(r"[A-Za-z0-9_]+\.(jpg|png)", name):
                self._file(os.path.join(STIM_DIR, name), "image/jpeg")
            else:
                self._json({"error": "bad name"}, 400)
        elif path == "/static/style.css":
            self._file(os.path.join(STATIC_DIR, "style.css"), "text/css; charset=utf-8")
        elif path == "/static/app.js":
            self._file(os.path.join(STATIC_DIR, "app.js"),
                       "application/javascript; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        if u.path == "/api/register":
            self._register(data)
        elif u.path == "/api/rate":
            self._rate(data)
        elif u.path == "/api/finish":
            self._finish(data)
        elif u.path == "/admin/reset":
            if self._admin_ok(u):
                self._reset(data)
        else:
            self._json({"error": "not found"}, 404)

    def _admin_ok(self, u) -> bool:
        q = parse_qs(u.query)
        return q.get("token", [""])[0] == ADMIN_TOKEN

    # ---------- API ----------
    def _register(self, data):
        consent = 1 if data.get("consent") else 0
        if not consent:
            self._json({"error": "consent required"}, 400)
            return
        pid = "P" + "".join(random.SystemRandom().choice(string.ascii_lowercase + string.digits)
                            for _ in range(10))
        demo = json.dumps({
            "age": data.get("age"), "gender": data.get("gender"),
            "vision_normal": data.get("vision_normal"),
            "expertise": data.get("expertise"),
            "display": data.get("display"), "device": data.get("device"),
        }, ensure_ascii=False)
        with _LOCK, db() as conn:
            conn.execute(
                "INSERT INTO participants(id, created_at, consent, demographics,"
                " acuity_ok, display) VALUES (?,?,?,?,?,?)",
                (pid, time.time(), 1, demo,
                 1 if data.get("acuity_ok") else 0,
                 data.get("display", "")))
        self._json({"participant": pid, "warmup": N_WARMUP})

    def _next(self, pid: str):
        with db() as conn:
            row = conn.execute(
                "SELECT finished, withdrawn FROM participants WHERE id=?",
                (pid,)).fetchone()
            if not row:
                self._json({"error": "unknown participant"}, 404)
                return
            if row[1]:
                self._json({"error": "withdrawn"}, 403)
                return
            done = {r[0] for r in conn.execute(
                "SELECT base||'|'||cond FROM ratings WHERE participant=?",
                (pid,)).fetchall()}
        trials = build_playlist(pid)
        remaining = [t for t in trials if f"{t['base']}|{t['cond']}" not in done]
        if not remaining:
            self._json({"done": True})
            return
        t = remaining[0]
        idx = trials.index(t)
        warmup = idx < N_WARMUP
        man = load_manifest()
        fmap = {s["base"]: s["files"] for s in man["stimuli"]}
        if t["base"] not in fmap:
            self._next(pid)  # skip malformed
            return
        cond = t["cond"].rstrip(REPEAT_TAG) or t["cond"]
        if cond not in fmap[t["base"]]:
            self._json({"error": "missing stimulus file"}, 500)
            return
        self._json({
            "done": False,
            "trial_index": idx,
            "n_total": len(trials),
            "warmup": warmup,
            "base": t["base"],
            "cond": cond,
            "is_repeat": t["is_repeat"],
            "url": "/stimuli/" + fmap[t["base"]][cond],
        })

    def _rate(self, data):
        pid = data.get("participant", "")
        rating = data.get("rating")
        mode = data.get("mode", "absolute")
        with _LOCK, db() as conn:
            conn.execute(
                "INSERT INTO ratings(participant, base, cond, is_repeat, rating,"
                " severity_delta, rt_ms, shown_left, trial_index, ts)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pid, data.get("base"), data.get("cond"),
                 1 if data.get("is_repeat") else 0,
                 int(rating) if rating is not None else None,
                 data.get("severity_delta"),
                 int(data.get("rt_ms", 0)),
                 data.get("shown_left"),
                 data.get("trial_index"), time.time()))
        self._json({"ok": True})

    def _finish(self, data):
        with _LOCK, db() as conn:
            conn.execute("UPDATE participants SET finished=1 WHERE id=?",
                         (data.get("participant", ""),))
        self._json({"ok": True})

    def _progress(self, pid: str):
        with db() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM ratings WHERE participant=?",
                (pid,)).fetchone()[0]
        trials = build_playlist(pid)
        self._json({"n_rated": n, "n_total": len(trials)})

    def _export_json(self):
        with db() as conn:
            parts = [dict(zip(("id", "created_at", "consent", "demographics",
                               "acuity_ok", "display", "finished", "withdrawn"), r))
                     for r in conn.execute("SELECT * FROM participants")]
            ratings = [dict(zip(("id", "participant", "base", "cond", "is_repeat",
                                 "rating", "severity_delta", "rt_ms", "shown_left",
                                 "trial_index", "ts"), r))
                       for r in conn.execute("SELECT * FROM ratings")]
        self._json({"participants": parts, "ratings": ratings,
                    "manifest": load_manifest()})

    def _export_csv(self):
        import csv as _csv
        import io
        buf = io.StringIO()
        w = _csv.writer(buf)
        w.writerow(["participant", "base", "cond", "is_repeat", "rating",
                    "severity_delta", "rt_ms", "shown_left", "trial_index", "ts"])
        with db() as conn:
            for r in conn.execute(
                    "SELECT participant, base, cond, is_repeat, rating,"
                    " severity_delta, rt_ms, shown_left, trial_index, ts"
                    " FROM ratings ORDER BY participant, trial_index"):
                w.writerow(r)
        body = buf.getvalue().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition",
                         "attachment; filename=eaqi_study_ratings.csv")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reset(self, data):
        pid = data.get("participant", "")
        with _LOCK, db() as conn:
            conn.execute("UPDATE participants SET withdrawn=1 WHERE id=?", (pid,))
            conn.execute("DELETE FROM ratings WHERE participant=?", (pid,))
        self._json({"ok": True})


ADMIN_TOKEN = "study2025"


def main() -> None:
    global ADMIN_TOKEN, BLOCK_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--admin-token", default="study2025")
    ap.add_argument("--block-size", type=int, default=240,
                    help="trials per observer (counterbalanced rotation); "
                         "0 = full 630-stimulus playlist")
    args = ap.parse_args()
    ADMIN_TOKEN = args.admin_token
    BLOCK_SIZE = args.block_size
    init_db()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"E-AQI study platform  ->  http://localhost:{args.port}")
    print(f"admin dashboard       ->  http://localhost:{args.port}/admin?token={ADMIN_TOKEN}")
    print(f"block size            ->  {BLOCK_SIZE or 'full playlist'}")
    print(f"stimuli dir           ->  {STIM_DIR}")
    print("Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
