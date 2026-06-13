"""CLI: size an experiment from the command line.

  # mean metric, with CUPED and a daily traffic estimate -> duration
  python -m ab_design --kind mean --baseline 10 --sd 5 --mde 0.1 \
      --rho 0.6 --daily 8000

  # proportion metric, clustered randomization, 3 primary metrics
  python -m ab_design --kind proportion --baseline 0.10 --mde 0.02 \
      --cluster-size 12 --icc 0.05 --k-primary 3
"""
import argparse

from .design import sample_size, allocate_alpha, format_design_report


def main():
    ap = argparse.ArgumentParser(description="Experiment design / sample-size calculator.")
    ap.add_argument("--kind", choices=["mean", "proportion"], required=True)
    ap.add_argument("--baseline", type=float, required=True,
                    help="control mean (kind=mean) or control rate (kind=proportion)")
    ap.add_argument("--mde", type=float, required=True, help="absolute effect to detect")
    ap.add_argument("--sd", type=float, default=None, help="metric std (kind=mean)")
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--one-sided", action="store_true")
    ap.add_argument("--rho", type=float, default=0.0, help="CUPED covariate correlation")
    ap.add_argument("--icc", type=float, default=None, help="intra-cluster correlation")
    ap.add_argument("--cluster-size", type=int, default=None, help="units per cluster")
    ap.add_argument("--k-primary", type=int, default=1,
                    help="number of primary metrics (Bonferroni-splits alpha)")
    ap.add_argument("--daily", type=float, default=None, help="daily users per arm -> duration")
    args = ap.parse_args()

    alpha = args.alpha
    if args.k_primary > 1:
        alpha = allocate_alpha(args.alpha, args.k_primary)[0]
        print(f"[alpha allocation] {args.k_primary} primary metrics -> "
              f"per-metric alpha = {alpha:.4f} (Bonferroni)\n")

    res = sample_size(
        args.kind, baseline=args.baseline, mde=args.mde, sd=args.sd,
        power=args.power, alpha=alpha, two_sided=not args.one_sided,
        rho=args.rho, icc=args.icc, cluster_size=args.cluster_size,
    )
    print(format_design_report(res, daily_users_per_arm=args.daily))


if __name__ == "__main__":
    main()
