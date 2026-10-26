# E-AQI Reference Implementation

Reference code for **"E-AQI: Edge-Aware Quality Index for Perceptual Clipping
Assessment in Enhanced Images"** (honest revision).

Status mirrors the revised manuscript: this is a **prototype** implementation
of the corrected formulation. It does NOT claim parameter/latency budgets for
a production model and does NOT include trained checkpoints. Numbers quoted in
the paper come only from actually-run scripts in this repository.

## Layout

| Path | Contents |
|---|---|
| `eatqi/clipping.py` | Two-sided adaptive saturation map (Sec. III-B/C, Eqs. 1–8) |
| `eatqi/entropy.py` | Windowed Shannon entropy via bin-indicator pooling (Sec. III-D, Eqs. 9–10) |
| `eatqi/surrogate.py` | Smooth histogram pile-up surrogate with LSE numerator (Sec. III-H, Eqs. 13–15) |
| `eatqi/model.py` | Prototype architecture: common H/4 lattice, cross-attention fusion, FPN head, aux heads (Sec. III-A/E/F) |
| `eatqi/losses.py` | Objective L_MSE + λr L_rank + λa L_aux, mask-aware, polarity utilities (Sec. III-G, Eq. 12) |
| `eatqi/metrics.py` | SRCC/KRCC, monotone 4PL calibration, PLCC/RMSE, paired source-grouped bootstrap, ≥3-seed aggregation (Sec. VI-C, Eq. 16) |
| `eatqi/stats.py` | Paired t, d_z with noncentral-t CI, exact power / min-n (Sec. V-C) |
| `eatqi/datasets.py` | Loaders for KonIQ-10k / SPAQ / PIPAL / LIVE / KADID-10k + source-disjoint splits (Sec. VI-A/B) |
| `eatqi/latency.py` | Profiling harness per Sec. VI-E reporting rules |
| `synthetic_checks.py` | Section IV / Table I checks (artificial values only) |
| `surrogate_diagnostics.py` | 0.192308/0.096154 reference-pair reproduction + fd derivative diagnostics |
| `smoke_model.py` | Forward/backward autograd audit through the full objective |
| `tests/run_tests.py` | 32 unit tests (all passing) |
| `clipping_location_study.py` | Computational study 1 (Sec. V-B): 168 area-matched E/T pairs, 4 IQMs -> `../results/clipping_location_study.json` |
| `enhancement_clip_response.py` | Computational study 2 (Sec. V-C): gamma sweep, clip dose vs IQM -> `../results/enhancement_clip_response.json` |
| `profile_gpu.py` | Sec. V-D GPU profile -> `../results/latency_gpu.json` |
| `train_koniq.py` | KonIQ-10k quality-transfer training (Sec. VI-B), 3 seeds -> `../results/koniq_eaqi_results.json` |
| `eval_baselines_koniq.py` | Training-free baselines + PCoE on the same test split -> `../results/koniq_baselines_seed0_test.json` |
| `data/` | Datasets (downloaded by the author; NOT committed): BSDS500 `clipstim/`, KonIQ-10k `koniq10k/` |

## Quick start

```bat
cd code
python -m tests.run_tests
python synthetic_checks.py                 :: writes ..\results\synthetic_checks.json
python surrogate_diagnostics.py
python smoke_model.py
python -m eatqi.latency --device cuda --precision fp16 --reps 200   :: on target GPU
```

## What is runnable NOW and EXECUTED (outputs in `../results/`)

* Table I synthetic checks — `python synthetic_checks.py`
* Surrogate limit/diagnostics — `python surrogate_diagnostics.py`
* Unit tests (32 passing) — `python -m tests.run_tests`
* Autograd smoke test — `python smoke_model.py`
* Computational study 1 — `python clipping_location_study.py --max-images 240`
  (BSDS500 base photographs in `data/clipstim/`; needs `pyiqa`)
* Computational study 2 — `python enhancement_clip_response.py --max-images 60`
* GPU profile — `python profile_gpu.py` (GTX 1050 Ti numbers in the paper)
* KonIQ-10k transfer training — `python train_koniq.py --epochs 12 --seeds 0 1 2`
  (dataset at `data/koniq10k/`, official release structure) — EXECUTED:
  SRCC 0.708±0.011, KRCC 0.516±0.011, PLCC 0.749±0.012, RMSE 0.122±0.002
  (mean ± SD over 3 seeds; per-seed values and fitted 4PL parameters in
  `../results/koniq_eaqi_results.json`)
* KonIQ baselines — `python eval_baselines_koniq.py --seed 0 --split test`
  — EXECUTED: NIQE 0.067, BRISQUE −0.020, PIQE −0.154, TOPIQ-NR 0.879,
  PCoE clip-fraction 0.191 (SRCC on the identical 2,015-image test partition;
  `../results/koniq_baselines_seed0_test.json`)

## Remaining author-side runs (prospective, marked as such in the paper)

1. SPAQ / PIPAL / LIVE / KADID-10k transfers (loaders in `eatqi/datasets.py`).
2. One LLIE-IQA release (SQUARE-LOL or LIEQ) for the independent validation.
3. Deeper baselines with their checkpoints (LIQE, CONTRIQUE, DEIQT, Q-Align,
   IACA, NLIEE) — report unavailable ones as unavailable.
4. Mobile-INT8 latency row (kept separate from desktop rows by protocol).
5. The preregistered human study (design in Sec. V of the paper).

## Ground rules encoded here

* No fabricated numbers anywhere; every JSON in `../results/` is script output.
* Fixed-image LPIPS term is excluded from the objective (zero-gradient issue).
* Significance stars are never recovered from SRCC gap magnitudes.
* Per-seed mean ± sample SD only with ≥ 3 seeds.
* Cross-dataset Δ_gen uses target-specific in-domain scores
  (Sec. VI-D definition).
