"""CLI: estimate the ATE, CATE, and a subgroup-fishing guard from a CSV.

  python -m ab_cate --data exp.csv --arm group --outcome revenue \
      --covariates pre_revenue,tenure_days,visits --learner x \
      --subgroup-col country
"""
import argparse

import numpy as np
import pandas as pd

from .cate import (lin_estimator, s_learner, t_learner, x_learner,
                   subgroup_fishing_guard, cate_summary, format_cate_report)

_LEARNERS = {"s": s_learner, "t": t_learner, "x": x_learner}


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def main():
    ap = argparse.ArgumentParser(description="Heterogeneous treatment effects (CATE).")
    ap.add_argument("--data", required=True, help="CSV path")
    ap.add_argument("--arm", required=True, help="assignment column (0/1 or two labels)")
    ap.add_argument("--outcome", required=True, help="outcome metric column")
    ap.add_argument("--covariates", required=True, help="comma-separated covariate columns")
    ap.add_argument("--learner", choices=["s", "t", "x"], default=None,
                    help="meta-learner for the CATE summary (optional)")
    ap.add_argument("--subgroup-col", default=None,
                    help="categorical column; each value becomes a fished subgroup")
    ap.add_argument("--ridge", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    labels = sorted(df[args.arm].unique())
    if len(labels) != 2:
        raise SystemExit(f"expected exactly 2 arms in '{args.arm}', got {labels}")
    T = (df[args.arm] == labels[-1]).astype(int).to_numpy()
    X = df[_split(args.covariates)].to_numpy(float)
    Y = df[args.outcome].to_numpy(float)

    lin = lin_estimator(X, T, Y)

    summ = None
    if args.learner:
        model = _LEARNERS[args.learner](X, T, Y, ridge=args.ridge)
        summ = cate_summary(model, X)

    guard = None
    if args.subgroup_col:
        subgroups = {f"{args.subgroup_col}={v}": (df[args.subgroup_col] == v).to_numpy()
                     for v in df[args.subgroup_col].unique()}
        guard = subgroup_fishing_guard(T, Y, subgroups)

    print(format_cate_report(lin=lin, cate_summary=summ, guard=guard))


if __name__ == "__main__":
    main()
