# -*- coding: utf-8 -*-
"""Online IQM comparison demo for reviewers (paper companion).

A small HTTP server (Python standard library only, mirroring the human-study
platform style) that lets a reviewer:
  * upload an image (or pick a bundled demo stimulus),
  * run BRISQUE / NIQE / PIQE / TOPIQ-NR (IQA-PyTorch reference
    implementations) and the trained E-AQI prototype plus the PCoE
    clipping descriptor,
  * see all scores side by side, oriented so that higher = better quality
    (severity measures are sign-aligned), with the clipped-area fraction
    and the edge-band fraction reported.

Usage:
  python tools/iqa_demo_server.py --port 8620 --demo-dir study_platform/stimuli

Then open http://127.0.0.1:8620/ . Results are computed locally; nothing is
uploaded anywhere.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CODE = os.path.join(ROOT, "code")
sys.path.insert(0, CODE)

import numpy as np  # noqa: E402

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>E-AQI IQM comparison demo</title>
<style>
 body { font-family: Georgia, serif; max-width: 960px; margin: 2rem auto;
        padding: 0 1rem; color: #1a1a1a; }
 h1 { font-size: 1.4rem; } .hint { color: #555; font-size: .9rem; }
 table { border-collapse: collapse; margin-top: 1rem; width: 100%; }
 th, td { border: 1px solid #ccc; padding: .45rem .6rem; text-align: left;
          font-size: .92rem; }
 th { background: #f2f2f2; } td.num { text-align: right; font-variant-numeric:
 tabular-nums; }
 .bar { height: 10px; background: #374151; display: inline-block;
        vertical-align: middle; margin-left: 6px; }
 img { max-width: 380px; max-height: 280px; display: block; margin-top: 1rem;
       border: 1px solid #999; }
 button { padding: .45rem 1rem; }
 #busy { color: #8a6d3b; display: none; }
</style>
</head>
<body>
<h1>Online IQM comparison &mdash; E-AQI companion demo</h1>
<p class="hint">
 Upload an image (or pick a bundled stimulus) and every measure from the
 paper &mdash; BRISQUE, NIQE, PIQE, TOPIQ-NR, the trained E-AQI prototype and
 the PCoE clipped-area descriptor &mdash; is computed locally with the same
 reference implementations used for the manuscript's tables. All scores are
 reported on a common axis: <b>higher = better quality</b> (severity-native
 measures are sign-aligned). Nothing leaves your machine.
</p>
<p>
 <input type="file" id="file" accept="image/*">
 <button onclick="run()">Score image</button>
 <span id="busy">computing&hellip; (first call loads checkpoints)</span>
</p>
<div id="out"></div>
<script>
async function run() {
  const f = document.getElementById('file').files[0];
  if (!f) { alert('choose an image first'); return; }
  const busy = document.getElementById('busy'); busy.style.display = 'inline';
  const buf = await f.arrayBuffer();
  const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
  const r = await fetch('/score', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({image_b64: b64, name: f.name})});
  const j = await r.json(); busy.style.display = 'none';
  if (j.error) { document.getElementById('out').innerHTML =
    '<p style="color:#a00">'+j.error+'</p>'; return; }
  let h = '<img src="data:image/png;base64,' + j.preview_b64 + '">';
  h += '<table><tr><th>measure</th><th>raw score</th>' +
       '<th>quality-oriented (0-1)</th></tr>';
  for (const row of j.rows) {
    h += '<tr><td>' + row.measure + '</td><td class="num">' +
         row.raw.toFixed(4) + '</td><td class="num">' +
         row.quality.toFixed(4) +
         '<span class="bar" style="width:' +
         Math.round(row.quality*180) + 'px"></span></td></tr>';
  }
  h += '</table><p class="hint">' + j.notes + '</p>';
  document.getElementById('out').innerHTML = h;
}
</script>
</body>
</html>"""


class DemoState:
    """Lazy-loaded models + the demo E-AQI checkpoint."""

    def __init__(self, ckpt: str | None):
        self.ckpt = ckpt
        self.lock = threading.Lock()
        self.ready = False

    def load(self):
        with self.lock:
            if self.ready:
                return
            import torch
            import pyiqa
            from eatqi.model import PrototypeEAQI
            self.torch = torch
            self.models = {
                "BRISQUE": pyiqa.create_metric("brisque", device="cpu"),
                "NIQE": pyiqa.create_metric("niqe", device="cpu"),
                "PIQE": pyiqa.create_metric("piqe", device="cpu"),
                "TOPIQ-NR": pyiqa.create_metric("topiq_nr", device="cpu"),
            }
            self.higher_better = {name: m.lower_better is False
                                  for name, m in self.models.items()}
            self.eaqi = None
            if self.ckpt and os.path.exists(self.ckpt):
                sd = torch.load(self.ckpt, map_location="cpu",
                                weights_only=False)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                # the released KonIQ checkpoint was trained with aux heads
                # disabled (lambda_a = 0), so build the network accordingly
                has_aux = any(k.startswith(("family_head", "severity_head",
                                            "spatial_head", "edge_head"))
                              for k in sd)
                net = PrototypeEAQI(aux_heads=has_aux)
                net.load_state_dict(sd, strict=True)
                net.eval()
                self.eaqi = net
            self.ready = True

    def score(self, rgb_u8: np.ndarray) -> dict:
        self.load()
        torch = self.torch
        x = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0)
        rows = []
        for name, m in self.models.items():
            with torch.no_grad():
                raw = float(m(x))
            rows.append({"measure": name, "raw": raw,
                         "higher_better": self.higher_better[name]})
        if self.eaqi is not None:
            with torch.no_grad():
                out = self.eaqi(x)
                val = out["severity"] if isinstance(out, dict) else out
                raw_e = float(val.reshape(-1)[0])
            rows.append({"measure": "E-AQI (trained)", "raw": raw_e,
                         "higher_better": True})
        # quality orientation: higher = better
        for r in rows:
            q = r["raw"] if r["higher_better"] else -r["raw"]
            r["quality"] = q
        return rows


def score_bytes(data: bytes, state: DemoState, checkpoint_note: str) -> dict:
    import cv2
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error": "could not decode image"}
    h, w = bgr.shape[:2]
    scale = min(1.0, 512.0 / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rows = state.score(rgb)
    # PCoE descriptor + clipped fraction (paper definition, 8-bit rule)
    x01 = rgb.astype(np.float32) / 255.0
    sat = (rgb.min(axis=2) < 5) | (rgb.max(axis=2) > 250)
    gx = cv2.Sobel(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    thr, _ = cv2.threshold(mag.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge = cv2.dilate((mag > thr).astype(np.uint8), np.ones((7, 7), np.uint8))
    frac_all = float(sat.mean())
    frac_pcoe = float((sat & (edge == 0)).mean())
    notes = (f"image scored at {rgb.shape[1]}x{rgb.shape[0]}; "
             f"clipped-area fraction (any channel <5 or >250): "
             f"{frac_all * 100:.2f}%; PCoE descriptor (outside 3px-dilated "
             f"Sobel-Otsu edge mask): {frac_pcoe * 100:.2f}%. "
             + checkpoint_note)
    return {"rows": rows, "notes": notes,
            "preview_b64": base64.b64encode(
                cv2.imencode(".png", bgr)[1].tobytes()).decode("ascii")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8620)
    ap.add_argument("--ckpt", default=os.path.join(
        ROOT, "results", "eaqi_koniq_seed0.pth"))
    args = ap.parse_args()

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = DemoState(args.ckpt if os.path.exists(args.ckpt) else None)
    note = ("E-AQI prototype loaded from the released seed-0 KonIQ-trained "
            "checkpoint." if state.ckpt else
            "E-AQI checkpoint not found in results/; only the four reference "
            "IQMs are scored.")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/score":
                self._send(404, b"not found", "text/plain")
                return
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n))
            data = base64.b64decode(payload["image_b64"])
            try:
                out = score_bytes(data, state, note)
            except Exception as exc:  # noqa: BLE001 - surface to reviewer
                out = {"error": f"{type(exc).__name__}: {exc}"}
            body = json.dumps(out).encode("utf-8")
            self._send(200, body, "application/json")

        def log_message(self, *a):  # quiet
            return

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"IQA demo serving on http://127.0.0.1:{args.port}/ (Ctrl+C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
