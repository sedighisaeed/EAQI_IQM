# -*- coding: utf-8 -*-
"""One-shot demo test writing results to a file (cmd-safe)."""
import base64
import glob
import io
import json
import os
import sys
import urllib.request

ROOT = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), ".."))

PORT = sys.argv[1] if len(sys.argv) > 1 else "8621"
cands = glob.glob(os.path.join(ROOT, "code", "data", "koniq10k",
                               "512x384", "*.jpg"))
if not cands:
    print("no demo image found")
    sys.exit(1)
b64 = base64.b64encode(open(sorted(cands)[0], "rb").read()).decode()

lines = []
try:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/score",
        data=json.dumps({"image_b64": b64}).encode(),
        headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    if "error" in out:
        lines.append("SERVER ERROR: " + out["error"])
        ok = False
    else:
        lines.append("notes: " + out["notes"])
        for r in out["rows"]:
            lines.append(f"{r['measure']:>16}: raw={r['raw']:+.4f} "
                         f"quality-oriented={r['quality']:+.4f}")
        ok = True
except Exception as exc:  # noqa: BLE001
    lines.append(f"CLIENT ERROR: {type(exc).__name__}: {exc}")
    ok = False

report = "\n".join(lines)
with io.open(os.path.join(ROOT, "tools", "_demo_test_out.txt"), "w",
             encoding="utf-8") as f:
    f.write(report + "\n")
print(report)
sys.exit(0 if ok else 1)
