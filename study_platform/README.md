# E-AQI Human Study Platform — Run Guide

Open-source online platform for the paired human perceptual study
(Section 5.4 of the manuscript). Python standard library only — no
dependencies to install.

## What is included

```
study_platform/
  server.py           HTTP server + SQLite storage (stdlib only)
  app.js              observer-facing front-end logic
  style.css           styles
  admin.html          observer/stimulus dashboard (token-protected)
  stimulus_generator.py  builds stimuli/ from BSDS500 (reuses the exact
                      computational-study-1 clipping pipeline)
  stimuli/            630 stimuli + manifest.json (126 base images x 5
                      conditions: E, T, X, N, C)
  study.db            created at first run (SQLite)
```

## 1. Generate stimuli (already executed; regenerate if needed)

```
cd study_platform
python stimulus_generator.py --bsds-root <path to BSDS500 images> --out stimuli
```

Executed output: 126 base images x 5 conditions = 630 stimuli
(180 candidates drawn from the same sorted BSDS500 train+val pool as
computational study 1; 54 candidates could not satisfy the area-matching
constraints and were excluded before any measurement).
`stimuli/manifest.json` records per-stimulus clipped-area fractions and the
area/histogram matching verification for each pair.

## 2. Start the server

```
python server.py --port 8000 --admin-token <your-token> [--block-size 240]
```

- Default `--block-size 240`: counterbalanced incomplete-block design. Each
  observer gets a deterministic random rotation of the full 756-trial
  playlist (630 + 20% repeats) truncated to 240 main trials + repeats
  (~24 min at 5 s view + 3 s rating). Across observers the rotations cover
  every pair; `analysis.py` enforces the preregistered minimum of 12
  ratings per base pair.
- `--block-size 0`: full single-session playlist (laboratory mode).

## 3. Observer flow (served at http://localhost:PORT/)

1. Informed consent + anonymous demographics
2. Display-adapted vision screening
3. Written instructions + 5 warm-up stimuli
4. 0-100 annoyance slider per stimulus (>= 2 s enforced viewing)
5. 20% within-observer repeats interleaved in the second half
6. Debrief + completion code

Every trial stores: stimulus id, condition (original or repeat tag), rating,
response time, and viewport metadata.

## 4. Admin dashboard

```
http://localhost:PORT/admin?token=<your-token>
```

Shows per-observer progress, median RT, repeat consistency, QC flags, and
per-pair rating counts.

## 5. Analysis (after data collection)

```
python analysis.py --db study.db --out ../results/human_study_analysis
```

Produces (as LaTeX tables + JSON):
- H1: per-participant paired T-E differences; two-sided Wilcoxon signed-rank;
  Hodges-Lehmann estimate; 2,000-resample bootstrap CI; Cohen's dz with
  noncentral-t CI
- H2: Friedman across E/T/X/N/C + Holm-corrected post-hoc Wilcoxon;
  crossed mixed-effects model (participant and base-image random effects,
  condition fixed effect) fit by projected/alternating least squares
- H3: optional 2AFC signed-severity test
- QC exclusions: < 80% trial coverage, median RT < 1.5 s, repeat consistency
  < 50% (agreement within +/-15 on the 0-100 scale)

## Manual (offline) data entry

If ratings are collected by hand instead of through the web platform, use
`manual_data_ratings.csv` / `manual_data_participants.csv` (pre-filled
templates; see `MANUAL_DATA_README.md` for the Persian guide), then:

```
python import_manual.py --dry-run   # validate
python import_manual.py             # write into study.db
```

## Ethics / recruitment notes

- The consent flow, privacy handling, and withdrawal text are embedded in
  the landing page; the timestamped protocol should be deposited before
  recruitment (preregistration).
- Target: >= 40 eligible observers (45-50 recruited to survive exclusions).
- Power (preregistered): N = 51 gives ~0.80 at alpha = 0.01 (two-sided) for
  dz = 0.5; N >= 40 gives ~0.72 at alpha = 0.05.
