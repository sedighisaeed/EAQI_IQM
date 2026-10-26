# -*- coding: utf-8 -*-
"""E-AQI: Edge-Aware Quality Index — reference implementation.

Modules mirror the revised manuscript sections:
  clipping.py   -> Section III-B/C  two-sided adaptive saturation map
  entropy.py    -> Section III-D    windowed local Shannon entropy
  surrogate.py  -> Section III-H    smooth histogram pile-up surrogate
  model.py      -> Section III-A/E/F  prototype architecture + fusion + heads
  losses.py     -> Section III-G    trainable objective + score polarity
  metrics.py    -> Section VI-C     SRCC/KRCC/PLCC/RMSE, 4PL, bootstrap, seeds
  stats.py      -> Section V-C      paired t, d_z, CIs, power analysis
  datasets.py   -> Section VI-A/B   dataset loaders + source-disjoint splits
  latency.py    -> Section VI-E     profiling harness

All numeric constants match the manuscript: eta = 0.015, alpha = 0.5 per
8-bit code value, default cutoffs 5 / 250, tightened cutoffs 25 / 230.
"""

__version__ = "0.2.0"

ETA = 0.015          # endpoint pile-up ratio threshold (Section III-B)
ALPHA_8BIT = 0.5     # sigmoid slope per code value (Section III-B)
DEFAULT_LO = 5.0     # 8-bit default lower (shadow) cutoff
DEFAULT_HI = 250.0   # 8-bit default upper (highlight) cutoff
TIGHT_LO = 25.0      # tightened lower cutoff when p_minus > ETA
TIGHT_HI = 230.0     # tightened upper cutoff when p_plus  > ETA
ENTROPY_BINS = 16    # luminance quantization bins (Section III-D)
ENTROPY_WINDOW = 9   # windowed entropy window (Section III-D)
