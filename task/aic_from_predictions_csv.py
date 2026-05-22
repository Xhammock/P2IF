#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从回归预测 CSV 计算高斯误差假定下的 AIC（与 compare_aic.py 中公式一致）。

    AIC = n * ln(RSS / n) + 2k

要求 CSV 至少包含真实值与预测值两列（默认列名 y_true, y_pred）。
应使用「拟合该下游模型时用的样本」上的预测（通常为 train_predictions.csv），
不要用测试集预测代替 in-sample AIC（除非你有意做别的分析并自行解释）。

参数 k
--------
- 线性回归：k = 系数个数（含截距）。
- LightGBM 等 boosting：没有公认的单一 k。本脚本支持：
  - 显式 --k（由你指定，便于论文中声明）；
  - --k-from-lightgbm-trees：从 LightGBM 文本模型中统计 Tree= 行数作为
    复杂度惩罚的粗近似（启发式，非教科书定义，仅在与同类设定对比时有参考意义）。

用法
----
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
        raise ValueError("需要 n > 0 且 RSS > 0")
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
            raise ValueError("CSV 无表头")
        if y_true_key not in reader.fieldnames or y_pred_key not in reader.fieldnames:
            raise ValueError(
                f"缺少列: 需要 {y_true_key!r}, {y_pred_key!r}；当前 {reader.fieldnames}"
            )
        for row in reader:
            yt = float(row[y_true_key])
            yp = float(row[y_pred_key])
            d = yt - yp
            rss += d * d
            n += 1
    if n == 0:
        raise ValueError("CSV 无数据行")
    return n, rss


def count_lightgbm_trees(model_txt_path: str) -> int:
    """LightGBM save_model 文本格式中每个根树一段，以 'Tree=i' 开头。"""
    tree_re = re.compile(r"^Tree=\d+\s*$")
    count = 0
    with open(model_txt_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if tree_re.match(line.strip()):
                count += 1
    if count == 0:
        raise ValueError(
            f"未在 {model_txt_path!r} 中解析到任何 Tree= 行，"
            "请确认是 LightGBM 导出的文本模型"
        )
    return count


def main() -> None:
    p = argparse.ArgumentParser(
        description="从 y_true/y_pred CSV 计算高斯回归 AIC（需指定或推断 k）"
    )
    p.add_argument("--csv", required=True, help="含 y_true、y_pred 的 CSV（建议 train_predictions）")
    p.add_argument("--y-true-col", default="y_true")
    p.add_argument("--y-pred-col", default="y_pred")
    k_group = p.add_mutually_exclusive_group(required=True)
    k_group.add_argument(
        "--k",
        type=int,
        help="参数个数 k（如线性模型为 特征数+截距；须自行声明含义）",
    )
    k_group.add_argument(
        "--k-from-lightgbm-trees",
        metavar="MODEL_TXT",
        help="用 LightGBM 文本模型中的树棵数作为 k（启发式）",
    )
    p.add_argument("--label", default="", help="打印时附带的模型名称标签")
    args = p.parse_args()

    n, rss = rss_from_csv(args.csv, args.y_true_col, args.y_pred_col)
    sigma2_hat = rss / n

    if args.k is not None:
        k = args.k
        k_source = f"用户指定 k={k}"
    else:
        k = count_lightgbm_trees(args.k_from_lightgbm_trees)
        k_source = f"LightGBM 树棵数 k={k}（文件 {args.k_from_lightgbm_trees!r}）"

    aic = aic_gaussian_linear(n, rss, k)
    name = args.label or args.csv

    print(f"标签: {name}")
    print(f"CSV: {args.csv}")
    print(f"n={n}, RSS={rss:.6g}, RSS/n (σ² MLE)={sigma2_hat:.6g}")
    print(f"{k_source}")
    print(f"AIC = n*ln(RSS/n) + 2k = {aic:.6f}")

    if args.k_from_lightgbm_trees:
        print(
            "\n说明: 用树棵数当 k 属于常见启发式，并非严格参数个数；"
            "跨方法（如与线性回归）比较 AIC 时需谨慎解释。",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
