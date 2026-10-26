# E-AQI — Companion Repository

Reference implementation, human-study platform, executed result files, and
manuscript source for:

**"E-AQI: Edge-Aware Quality Index for Perceptual Clipping Assessment in
Enhanced Images"** — Ali Hosseini, Saeed Sedighi (Islamic Azad University,
Shiraz). Submitted to *Journal of Visual Communication and Image
Representation*.

## What is here

| Directory | Contents |
|---|---|
| `code/` | Reference implementation (`eatqi/` package + study/training scripts). Every number in the paper comes from one of these scripts. |
| `study_platform/` | The paired human-study platform (server, consent/screening flow, admin dashboard, stimulus generator, analysis script), the anonymized ratings of the executed 51-observer study (`exported_ratings.csv`), and the per-stimulus manifest. |
| `results/` | Verbatim JSON outputs of every executed study cited in the paper. |
| `manuscript/latex/` | Full LaTeX source of the submission (`main.tex`, `refs.bib`, figures). |
| `tools/` | Aggregation/figure scripts and the **online IQM comparison demo** (`iqa_demo_server.py`). |
| `docs/review_guide.md` | Claim → source map and one-command verification instructions. |

## Quick verification (reviewers)

```bat
cd code && python -m tests.run_tests && python synthetic_checks.py
python tools/iqa_demo_server.py --port 8620   :: live IQM comparison in the browser
```

See `docs/review_guide.md` for the complete claim-by-claim map.

## Datasets

KonIQ-10k and BSDS500 are public downloads (see `docs/review_guide.md` §6);
they are **not** committed here. No other data are required to run the
synthetic checks, the IQM demo, or the human-study analysis.

## Human-study data

The executed study released here comprises 51 observers × 630 trials
(32,130 ratings) on the five matched clipping conditions of the paper.
Ratings are anonymized; the platform records only age band, gender,
self-reported display/device, and the ratings themselves. The analysis
script rebuilds every table in §V-D from the CSV alone.

## License

MIT (see `LICENSE`). Datasets remain under their own licenses.
