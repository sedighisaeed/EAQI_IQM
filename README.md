# EAQI_IQM — IQM Comparison Demo & Human-Study Platform

Companion repository for the E-AQI paper. It contains the two tools a
reviewer needs to interact with the work:

1. **`tools/iqa_demo_server.py`** — an online IQM comparison demo
2. **`study_platform/`** — the paired human-study (review) platform with the
   anonymized 51-observer rating data

No manuscript files are hosted here.

## 1. Online IQM comparison demo

Scores any uploaded image with the same reference implementations used in the
paper: BRISQUE, NIQE, PIQE, TOPIQ-NR (IQA-PyTorch), the trained E-AQI
prototype (released seed-0 KonIQ checkpoint in `results/`), plus the PCoE
clipped-area descriptor. Everything runs locally; nothing is uploaded.

```bat
pip install torch pyiqa opencv-python numpy
python tools/iqa_demo_server.py --port 8620
:: open http://127.0.0.1:8620/
```

## 2. Human-study platform (review system)

The full flow used for the paper's perceptual study: informed consent,
demographics, acuity screening, randomized 0–100 annoyance ratings over the
630 stimuli (126 base images × 5 matched conditions), admin dashboard, and
the analysis script.

```bat
cd study_platform
python server.py          :: serves the study at http://127.0.0.1:8000
python analysis.py        :: regenerates every study statistic from the data
```

`study_platform/exported_ratings.csv` holds the anonymized ratings of the
executed 51-observer study (32,130 trials). `analysis.py` re-derives all
reported statistics (Wilcoxon signed-rank, Hodges–Lehmann with bootstrap CI,
Cohen's d_z with noncentral-t CI, Friedman + Holm post-hoc) directly from
that CSV/database — no other data are required.

## Layout

| Path | Contents |
|---|---|
| `tools/iqa_demo_server.py` | online IQM comparison web demo |
| `study_platform/` | review platform: server, UI, stimulus manifest, ratings, analysis |
| `code/eatqi/` | E-AQI reference implementation (used by the demo) |
| `results/eaqi_koniq_seed0.pth` | trained E-AQI checkpoint loaded by the demo |

## License

MIT (see `LICENSE`).
