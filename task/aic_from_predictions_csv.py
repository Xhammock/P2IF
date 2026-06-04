#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Gaussian-error AIC from a regression prediction CSV (same formula as compare_aic.py).

    AIC = n * ln(RSS / n) + 2k

The CSV must contain at least true and predicted value columns (default: y_true, y_pred).
Use predictions on the samples used to fit the downstream model (typically train_predictions.csv).
Do not substitute test-set predictions for in-sample AIC unless you intend a different analysis and interpret it yourself.

Parameter k
-----------
- Linear regression: k = number of coefficients (including intercept).
- LightGBM and other boosting: no single universally accepted k. This script supports:
  - explicit --k (you specify it, e.g. for papers);
  - --k-from-lightgbm-trees: count Tree= lines in the LightGBM text model as a coarse
    complexity penalty (heuristic, not textbook definition; useful only when comparing like setups).

Usage
-----
  python task/aic_from_predictions_csv.py \\
    --csv results/train_20251222_220544/vitality_prediction/train_predictions.csv \\
    --k-from-lightgbm-trees results/train_20251222_220544/vitality_prediction/lightgbm_model.txt

  python task/aic_from_predictions_csv.py --csv path/to/train_predictions.csv --k 130
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from typing import Tuple


def aic_gaussian_linear(n: int, rss: float, k: int) -> float:
    if n <= 0 or rss <= 0:
        raise ValueError("requires n > 0 and RSS > 0")
    return float(n * math.log(rss / n) + 2 * k)


def rss_from_csv(
    path: str,
    y_true_key: str = "y_true",
    y_pred_key: str = "y_pred",
) -> Tuple[int, float]:
    n = 0
    rss = 0.0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        if y_true_key not in reader.fieldnames or y_pred_key not in reader.fieldnames:
            raise ValueError(
                f"Missing columns: need {y_true_key!r}, {y_pred_key!r}; got {reader.fieldnames}"
            )
        for row in reader:
            yt = float(row[y_true_key])
            yp = float(row[y_pred_key])
            d = yt - yp
            rss += d * d
            n += 1
    if n == 0:
        raise ValueError("CSV has no data rows")
    return n, rss


def count_lightgbm_trees(model_txt_path: str) -> int:
    """In LightGBM save_model text format, each root tree starts with 'Tree=i'."""
    tree_re = re.compile(r"^Tree=\d+\s*$")
    count = 0
    with open(model_txt_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if tree_re.match(line.strip()):
                count += 1
    if count == 0:
        raise ValueError(
            f"No Tree= lines parsed in {model_txt_path!r}; "
            "confirm this is a LightGBM exported text model"
        )
    return count


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute Gaussian regression AIC from y_true/y_pred CSV (k must be specified or inferred)"
    )
    p.add_argument("--csv", required=True, help="CSV with y_true, y_pred (train_predictions recommended)")
    p.add_argument("--y-true-col", default="y_true")
    p.add_argument("--y-pred-col", default="y_pred")
    k_group = p.add_mutually_exclusive_group(required=True)
    k_group.add_argument(
        "--k",
        type=int,
        help="Parameter count k (e.g. features+intercept for linear; declare meaning yourself)",
    )
    k_group.add_argument(
        "--k-from-lightgbm-trees",
        metavar="MODEL_TXT",
        help="Use tree count in LightGBM text model as k (heuristic)",
    )
    p.add_argument("--label", default="", help="Model name label for printing")
    args = p.parse_args()

    n, rss = rss_from_csv(args.csv, args.y_true_col, args.y_pred_col)
    sigma2_hat = rss / n

    if args.k is not None:
        k = args.k
        k_source = f"user-specified k={k}"
    else:
        k = count_lightgbm_trees(args.k_from_lightgbm_trees)
        k_source = f"LightGBM tree count k={k} (file {args.k_from_lightgbm_trees!r})"

    aic = aic_gaussian_linear(n, rss, k)
    name = args.label or args.csv

    print(f"Label: {name}")
    print(f"CSV: {args.csv}")
    print(f"n={n}, RSS={rss:.6g}, RSS/n (sigma^2 MLE)={sigma2_hat:.6g}")
    print(f"{k_source}")
    print(f"AIC = n*ln(RSS/n) + 2k = {aic:.6f}")

    if args.k_from_lightgbm_trees:
        print(
            "\nNote: Using tree count as k is a common heuristic, not a strict parameter count; "
            "interpret cross-method AIC comparisons (e.g. vs linear regression) with care.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
