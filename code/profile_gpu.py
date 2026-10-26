# -*- coding: utf-8 -*-
"""Profile the E-AQI prototype (Section VI-E protocol):
parameters (total / trainable), latency distribution on GPU, and peak memory.
Measured values feed the manuscript's efficiency discussion.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eatqi.model import PrototypeEAQI  # noqa: E402


def bench(model, x, warmup=10, rep=50):
    device = x.device
    # latency
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(rep):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mean_ms = (time.perf_counter() - t0) * 1000.0 / rep

        # peak memory
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            model(x)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak = None
    return mean_ms, peak


def main():
    out = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"device": torch.cuda.get_device_name(0)
               if device.type == "cuda" else "cpu",
               "torch": torch.__version__,
               "precision": "FP32",
               "batch": 1}

    for size in [(384, 512), (192, 256)]:
        h, w = size
        model = PrototypeEAQI(aux_heads=True).to(device).eval()
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x = torch.rand(1, 3, h, w, device=device)
        mean_ms, peak = bench(model, x)
        key = f"{w}x{h}"
        results[key] = {
            "params_total_M": round(n_total / 1e6, 4),
            "params_trainable_M": round(n_train / 1e6, 4),
            "latency_mean_ms": round(mean_ms, 2),
            "peak_mem_MiB": round(peak, 1) if peak else None,
            "resolution_note": "input W x H",
        }
        print(key, results[key])

    os.makedirs(os.path.join("..", "results"), exist_ok=True)
    path = os.path.join("..", "results", "latency_gpu.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved ->", os.path.normpath(path))


if __name__ == "__main__":
    main()
