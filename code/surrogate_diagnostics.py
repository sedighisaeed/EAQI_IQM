# -*- coding: utf-8 -*-
"""Diagnostics for the smooth histogram surrogate on the exact Section IV
example: twenty values at 255, none at zero, deliberately nonuniform interior.
This reproduces the 0.192308 / 0.096154 reference pair quoted in the
manuscript's Table I (hard max-endpoint ratio vs superseded average).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eatqi.surrogate import (endpoint_average_ratio, hard_pileup_ratio,  # noqa
                             smooth_histogram, smooth_pileup_ratio)


def reference_sequence():
    """The Section IV/Table I sequence: 20x255, 0x0, structured interior.
    Interior engineered so the modal bin has 104 members, giving
    20/104 = 0.192308 for the hard max-endpoint ratio."""
    rng = np.random.default_rng(2024)
    interior = []
    for _ in range(104):
        interior.append(128)              # modal bin: code 128
    interior += list(rng.integers(1, 127, size=380))
    interior += list(rng.integers(129, 255, size=496))
    seq = np.array(interior + [255] * 20, dtype=np.float64)
    return seq


def main():
    seq = reference_sequence()
    hist = np.histogram(seq, bins=256, range=(0, 255))[0]
    print("N =", seq.size, " h(0) =", hist[0], " h(255) =", hist[-1],
          " modal =", hist.max())
    hard = hard_pileup_ratio(seq)
    old = endpoint_average_ratio(seq)
    print(f"hard max-endpoint ratio : {hard:.6f}")
    print(f"endpoint-average (old)  : {old:.6f}")
    print(f"factor-of-two check     : |2*old - hard| = {abs(2*old-hard):.2e}")
    for delta in (0.1, 0.25, 0.5):
        sm = smooth_pileup_ratio(seq, delta=delta, t=1e-5)
        print(f"smooth ratio (delta={delta:<4}, t=1e-5): {sm:.6f}")
    # derivative diagnostic on the six-value example
    vals = np.array([12.0, 37.0, 64.0, 128.0, 200.0, 251.0])
    h = 1e-4
    for i0 in (2, 3):
        vp, vm = vals.copy(), vals.copy()
        vp[i0] += h
        vm[i0] -= h
        d = (smooth_pileup_ratio(vp, delta=0.5, t=1e-5)
             - smooth_pileup_ratio(vm, delta=0.5, t=1e-5)) / (2 * h)
        print(f"fd derivative wrt value[{i0}]={vals[i0]:.0f}: {d:+.6f}")


if __name__ == "__main__":
    main()
