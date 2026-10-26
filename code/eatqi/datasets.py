# -*- coding: utf-8 -*-
"""Dataset loaders and source-disjoint split utilities (Sections VI-A/B).

The loaders assume the datasets have been downloaded by the author into
`data/` with their official release structures.  Nothing here fabricates
MOS labels: targets come only from the official annotation files.

Supported (documented in Section VI-B of the revised manuscript):
  KonIQ-10k, SPAQ, PIPAL, LIVE (quality transfer task),
  KADID-10k (pretraining resource), and an LLIE-IQA dataset hook
  (e.g., SQUARE-LOL / LIEQ / RLIE) with polarity documented per release.

Split rule (Section VI-A): all derivatives of a source image stay in a
single partition (source-disjoint splits).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np

DATA_ROOT = os.environ.get("EAQI_DATA", os.path.join("data"))


@dataclass
class ImageScore:
    path: str
    score: float          # dataset-native polarity (documented per dataset)
    source_id: str        # identity used for source-disjoint grouping


def load_koniq10k(root: str | None = None) -> list[ImageScore]:
    """KonIQ-10k: MOS in [1, 5], higher is better (Hosu et al., TIP 2020)."""
    root = root or os.path.join(DATA_ROOT, "koniq10k")
    meta = os.path.join(root, "koniq10k_distributions_sets.csv")
    out = []
    with open(meta, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["file_name"]
            out.append(ImageScore(
                path=os.path.join(root, "512x384", name),
                score=float(row["MOS_zscore"]),
                source_id=name,
            ))
    return out


def load_spaq(root: str | None = None) -> list[ImageScore]:
    """SPAQ: MOS in [1, 100], higher is better (Fang et al., CVPR 2020)."""
    root = root or os.path.join(DATA_ROOT, "SPAQ")
    meta = os.path.join(root, "Annotations spreadsheet",
                        "SPAQ_trainset_annotations.xlsx")
    # XLSX parsing intentionally avoided; convert once to CSV with:
    #   python tools/xlsx_to_csv.py  (kept in tools/)
    meta_csv = os.path.splitext(meta)[0] + ".csv"
    out = []
    with open(meta_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["Image name"]
            out.append(ImageScore(
                path=os.path.join(root, "TestImage_2560", name),
                score=float(row["MOS"]),
                source_id=name,
            ))
    return out


def load_pipal(root: str | None = None) -> list[ImageScore]:
    """PIPAL: Elo-derived scores, higher is better (Gu et al., ECCV 2020)."""
    root = root or os.path.join(DATA_ROOT, "PIPAL")
    meta = os.path.join(root, "Train_Score.txt")
    out = []
    with open(meta, encoding="utf-8") as f:
        for line in f:
            name, score = line.split()
            out.append(ImageScore(path=os.path.join(root, "Train", name),
                                  score=float(score), source_id=name))
    return out


def load_live(root: str | None = None) -> list[ImageScore]:
    """LIVE IQA: DMOS in [0, 100], HIGHER IS WORSE (Sheikh et al., TIP 2006).
    Polarity: keep severity orientation after documented normalization
    (Section III-G); do NOT silently flip."""
    root = root or os.path.join(DATA_ROOT, "live")
    meta = os.path.join(root, "data", "matlab", "data.mat")
    raise FileNotFoundError(
        f"Load LIVE DMOS from {meta} with scipy.io.loadmat in the author's "
        "environment; DMOS polarity handling must remain explicit.")


def load_kadid10k(root: str | None = None) -> list[ImageScore]:
    """KADID-10k: MOS in [1, 5], higher is better (Lin et al., QoMEX 2019).
    NOTE: KADIS-700k is a distinct UNLABELED resource — never substitute it
    (Section VI-A)."""
    root = root or os.path.join(DATA_ROOT, "kadid10k")
    meta = os.path.join(root, "kadid10k.csv")
    out = []
    with open(meta, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(ImageScore(
                path=os.path.join(root, "images", row["distorted_img"]),
                score=float(row["dmos"]),
                source_id=row["reference_img"],
            ))
    return out


# --------------------------------------------------------------------------
# source-disjoint splits (Section VI-A)
# --------------------------------------------------------------------------

def source_disjoint_splits(items: list[ImageScore],
                           train_frac: float = 0.7,
                           calib_frac: float = 0.1,
                           seed: int = 0,
                           ) -> dict[str, list[ImageScore]]:
    """Split by SOURCE identity so all derivatives of a source image land in
    one partition.  calib partition is used for 4PL calibration fitting
    (Section VI-C); test is disjoint from both."""
    rng = np.random.default_rng(seed)
    sources = sorted({it.source_id for it in items})
    perm = rng.permutation(len(sources))
    n_train = int(len(sources) * train_frac)
    n_calib = int(len(sources) * calib_frac)
    train_src = {sources[i] for i in perm[:n_train]}
    calib_src = {sources[i] for i in perm[n_train:n_train + n_calib]}
    parts: dict[str, list[ImageScore]] = {"train": [], "calib": [], "test": []}
    for it in items:
        if it.source_id in train_src:
            parts["train"].append(it)
        elif it.source_id in calib_src:
            parts["calib"].append(it)
        else:
            parts["test"].append(it)
    return parts


def group_ids(items: list[ImageScore]) -> np.ndarray:
    """Group identity per item (for the source-grouped bootstrap)."""
    return np.array([it.source_id for it in items])


def iter_batches(items: list[ImageScore], batch_size: int
                 ) -> Iterator[list[ImageScore]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
