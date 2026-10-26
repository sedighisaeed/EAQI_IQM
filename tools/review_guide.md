# Reviewer Guide — verifying the paper's numbers from this repository

This guide maps **every quantitative claim in the manuscript** to the file or
command that produces it. Everything runs offline from this repository; the
only downloads are the public datasets themselves.

## 1. One-command sanity check

```bat
cd code
python -m tests.run_tests          :: 32 unit tests
python synthetic_checks.py         :: rewrites ../results/synthetic_checks.json
python surrogate_diagnostics.py
python smoke_model.py
```

## 2. Claim → source map

| Paper claim (section) | Source file / command |
|---|---|
| Table 1 synthetic operator checks (§IV) | `results/synthetic_checks.json` ← `code/synthetic_checks.py` |
| Table 2 clipping-location sensitivity, 168 pairs (§V-A) | `results/clipping_location_study.json` ← `code/clipping_location_study.py` |
| E-AQI/PCoE rows of Table 2 (§V-A) | `results/eaqi_studies12.json` ← `code/eval_eaqi_studies.py` |
| Dose-response SRCCs, 480 levels (§V-B) | `results/enhancement_clip_response.json` ← `code/enhancement_clip_response.py` |
| Latency 20.9 ms / 226.6 ms, params 433,380 (§V-C) | `results/latency_gpu.json`, `results/latency_cpu.json` ← `code/profile_gpu.py` |
| KonIQ transfer row E-AQI 0.708±0.011 (Table 3, §VI-B) | `results/koniq_eaqi_results.json` ← `code/train_koniq.py --epochs 12 --seeds 0 1 2` |
| Baseline rows NIQE/BRISQUE/PIQE/TOPIQ-NR/PCoE (Table 3) | `results/koniq_baselines_seed0_test.json` ← `code/eval_baselines_koniq.py --seed 0 --split test` |
| Ablation Table 4 (§VI-F) | `results/koniq_ablation.json` ← `code/ablate_koniq.py --epochs 3` |
| Human study H1 table (§V-D Results) | `results/human_study_analysis.json` ← `study_platform/analysis.py` (DB rebuilt from `study_platform/exported_ratings.csv`) |
| Human study per-condition table (§V-D) | same JSON, `conditions` block |

## 3. Reproducing the human-study analysis from the released ratings

```bat
cd study_platform
python import_manual.py --ratings exported_ratings.csv --participants manual_data_participants.csv --db rebuilt.db
python analysis.py --db rebuilt.db
```

`analysis.py` re-derives every number in the two human-study tables
(Hodges–Lehmann, bootstrap CI, Cohen's d_z with noncentral-t CI, Friedman +
Holm post-hoc) and writes the LaTeX fragments. The anonymized ratings
(51 observers × 630 trials) are in `study_platform/exported_ratings.csv`;
no identifying observer information was collected.

## 4. Running the platform end-to-end (optional)

```bat
cd study_platform
python make_stimuli.py            :: regenerate the 630 stimuli (needs BSDS500)
python server.py                  :: serves the study at http://127.0.0.1:8000
```

Admin dashboard (progress, export): `http://127.0.0.1:8000/admin`.

## 5. Online IQM comparison demo

```bat
python tools/iqa_demo_server.py --port 8620
:: open http://127.0.0.1:8620/
```

Upload any image; BRISQUE / NIQE / PIQE / TOPIQ-NR (IQA-PyTorch reference
implementations) and the trained E-AQI prototype (released seed-0 checkpoint)
score it locally, sign-aligned so higher = better quality, with the
clipped-area fraction and the PCoE descriptor reported. This reproduces the
comparison logic of Tables 2–3 on arbitrary inputs, entirely offline.

## 6. Datasets (not included; download separately)

* KonIQ-10k official release → `code/data/koniq10k/`
* BSDS500 → `code/data/BSR/` (study-1 base photographs in `code/data/clipstim/`)

## 7. Ground rules encoded in this release

1. Every JSON in `results/` is verbatim script output; no hand-edited numbers.
2. Classical IQMs are never re-tuned per dataset (protocol §VI-C of the paper).
3. PLCC/RMSE only after monotone 4PL calibration on a disjoint partition.
4. Human repeats analyzed as repeats, never as independent observers.
5. The preregistered QC rules were fixed before recruitment and unchanged.
