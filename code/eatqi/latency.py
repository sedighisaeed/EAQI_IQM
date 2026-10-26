# -*- coding: utf-8 -*-
"""Latency and memory profiling harness (Section VI-E requirements).

Reports, per the manuscript's own rules:
  * total and TRAINABLE parameters separately, including frozen modules;
  * input resolution, precision, batch size, device, software stack;
  * synchronized timing with warm-up and repetitions, reported as
    distribution (median / IQR), never as a single best-case number;
  * desktop FP16 and mobile INT8 timings are NEVER combined into one claim.

Run on the actual target hardware before quoting any latency in the paper:
    python -m eatqi.latency --device cuda --precision fp16 --reps 200
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import torch

from .model import PrototypeEAQI


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "buffers": int(buffers),
        "total_params_M": round(total / 1e6, 3),
        "fp32_weight_storage_MB_dec": round(total * 4 / 1e6, 2),
    }


def benchmark_latency(model: torch.nn.Module, input_size=(1, 3, 512, 512),
                      device: str = "cpu", precision: str = "fp32",
                      warmup: int = 20, reps: int = 100,
                      batch_size: int = 1) -> dict:
    dev = torch.device(device)
    model = model.to(dev).eval()
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[precision]
    x = torch.rand(*input_size, device=dev)
    if batch_size != input_size[0]:
        x = x.repeat(batch_size, 1, 1, 1)
    x = x.to(dtype)

    with torch.no_grad():
        for _ in range(warmup):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            model(x)
        times = []
        for _ in range(reps):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1e3
    return {
        "device": str(dev),
        "device_name": (torch.cuda.get_device_name(0)
                        if dev.type == "cuda" else platform.processor()),
        "precision": precision,
        "batch_size": batch_size,
        "input_size": list(input_size),
        "repetitions": reps,
        "warmup": warmup,
        "latency_ms_median": float(np.median(times_ms)),
        "latency_ms_iqr": [float(np.percentile(times_ms, 25)),
                           float(np.percentile(times_ms, 75))],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--precision", default="fp32",
                    choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default="results/latency.json")
    args = ap.parse_args()

    model = PrototypeEAQI()
    params = count_parameters(model)
    lat = benchmark_latency(model, device=args.device,
                            precision=args.precision,
                            warmup=args.warmup, reps=args.reps,
                            batch_size=args.batch)
    report = {
        "parameters": params,
        "latency": lat,
        "note": ("Mobile INT8 timing must be measured separately on the "
                 "target device and reported as a distinct row; it must not "
                 "be merged with this desktop measurement (Section VI-E)."),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
