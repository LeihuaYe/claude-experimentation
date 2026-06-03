"""CLI: read out an experiment from a CSV.

  python -m ab_readout --data exp.csv --arm group --primary revenue \
      --metrics revenue,engagement,conversion --binary conversion \
      --guardrails latency,tickets --covariate pre_revenue
"""
import argparse

import pandas as pd

from .readout import run_readout, format_report


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def main():
    ap = argparse.ArgumentParser(description="Trustworthy A/B test readout.")
    ap.add_argument("--data", required=True, help="CSV path")
    ap.add_argument("--arm", required=True, help="assignment column (exactly 2 groups)")
    ap.add_argument("--primary", required=True, help="primary metric column")
    ap.add_argument("--metrics", default="", help="comma-separated metric columns (primary auto-included)")
    ap.add_argument("--binary", default="", help="comma-separated metrics that are 0/1")
    ap.add_argument("--guardrails", default="", help="comma-separated guardrail metrics")
    ap.add_argument("--covariate", default=None, help="pre-experiment covariate column for CUPED")
    ap.add_argument("--srm-alpha", type=float, default=0.001)
    ap.add_argument("--fdr-alpha", type=float, default=0.05)
    ap.add_argument("--expected-ratio", type=float, default=0.5, help="expected control share")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    res = run_readout(
        df, arm_col=args.arm, primary=args.primary,
        metrics=_split(args.metrics) or [args.primary],
        binary=_split(args.binary), guardrails=_split(args.guardrails),
        covariate=args.covariate, srm_alpha=args.srm_alpha,
        fdr_alpha=args.fdr_alpha, expected_ratio=args.expected_ratio,
    )
    print(format_report(res))


if __name__ == "__main__":
    main()
