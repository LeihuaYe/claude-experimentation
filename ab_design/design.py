"""Experiment design: power / MDE / sample-size / duration as one quadrilateral.

Most underpowered tests are decided before they start. Sizing is four numbers in
tension — fix any three and the fourth follows:

  sample size  <->  MDE (minimum detectable effect)  <->  power  <->  significance

This module computes any leg from the others, for both mean and proportion metrics,
and handles the design decisions that actually move required n:
  - CUPED            — a pre-experiment covariate (corr rho) cuts variance ~rho^2,
                       so it cuts required n by (1 - rho^2). Size for it up front.
  - clustering       — when the randomization unit holds many analysis units
                       (sessions in a user), variance inflates by the design effect
                       1 + (m-1)*ICC. Sizing on raw n is the classic power mistake.
  - ratio metrics    — clicks/impression etc. are ratios of two random sums; their
                       variance needs the delta method, not the naive sample variance.
  - many primaries   — testing k primary metrics spends alpha k ways; size for the
                       corrected alpha or you are underpowered at decision time.
  - switchback       — when the unit is a *time period*, power is driven by the number
                       of periods, not the number of users.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _zab(alpha=0.05, power=0.80, two_sided=True):
    """Critical z for the significance level and the power."""
    z_a = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    return float(z_a), float(z_b)


def cluster_design_effect(cluster_size, icc):
    """Variance inflation when randomizing clusters of `cluster_size` with
    intra-cluster correlation `icc`. n_effective = n_naive * design_effect."""
    return 1.0 + (cluster_size - 1) * icc


def allocate_alpha(alpha, k, weights=None):
    """Split a family-wise alpha across k primary metrics. Equal (Bonferroni) by
    default; pass weights to spend more alpha on the metrics you care most about."""
    if weights is None:
        return [alpha / k] * k
    w = np.asarray(weights, float)
    return (alpha * w / w.sum()).tolist()


def ratio_variance(numerator, denominator):
    """Delta-method per-unit variance of the ratio estimator R = mean(num)/mean(den).

    Var(R) ~= (1/mu_d^2) * (Var(num) - 2*R*Cov(num,den) + R^2*Var(den)). Returns the
    per-unit variance you then feed to sample_size(kind="mean", sd=sqrt(var))."""
    num = np.asarray(numerator, float)
    den = np.asarray(denominator, float)
    mu_n, mu_d = num.mean(), den.mean()
    R = mu_n / mu_d
    cov = np.cov(num, den, ddof=1)
    var = (cov[0, 0] - 2 * R * cov[0, 1] + R ** 2 * cov[1, 1]) / mu_d ** 2
    return {"ratio": float(R), "unit_variance": float(var), "unit_sd": float(np.sqrt(var))}


def switchback_design(n_periods, seed=0, block=1):
    """Assign `n_periods` time buckets to control/treatment, balanced, in blocks of
    `block` consecutive periods (longer blocks reduce switching but cost effective n).
    The unit of inference is the PERIOD: power scales with n_periods, not users."""
    if n_periods % block != 0:
        raise ValueError(f"n_periods={n_periods} not divisible by block={block}")
    n_blocks = n_periods // block
    rng = np.random.default_rng(seed)
    half = n_blocks // 2
    labels = np.array([0] * half + [1] * (n_blocks - half))
    rng.shuffle(labels)
    assign = np.repeat(labels, block)
    return {"assignment": assign.tolist(), "n_periods": n_periods, "block": block,
            "n_control_periods": int((assign == 0).sum()),
            "n_treat_periods": int((assign == 1).sum())}


# ----------------------------------------------------------------------------- #
#  The quadrilateral: sample_size <-> mde <-> power, for means and proportions.  #
# ----------------------------------------------------------------------------- #

def _se_unit_mean(sd, rho):
    """Per-arm, per-unit standard deviation after CUPED variance reduction."""
    return sd * np.sqrt(1 - rho ** 2)


def sample_size(kind, baseline, mde, sd=None, power=0.80, alpha=0.05,
                two_sided=True, rho=0.0, icc=None, cluster_size=None):
    """Per-arm sample size to detect `mde` at the given power and significance.

    kind="mean": needs `sd` (std of the metric); `mde` is the absolute mean difference.
    kind="proportion": `baseline` is the control rate; `mde` is the absolute lift.
    rho: CUPED covariate correlation (cuts n by 1-rho^2, means only).
    icc + cluster_size: inflate by the cluster design effect and report clusters/arm.
    """
    z_a, z_b = _zab(alpha, power, two_sided)
    if kind == "mean":
        if sd is None:
            raise ValueError("kind='mean' requires sd")
        sd_eff = _se_unit_mean(sd, rho)
        n = (z_a + z_b) ** 2 * 2 * sd_eff ** 2 / mde ** 2
    elif kind == "proportion":
        p1 = baseline
        p2 = baseline + mde
        if not (0 < p1 < 1 and 0 < p2 < 1):
            raise ValueError(f"proportion out of range: p1={p1}, p2={p2}")
        pbar = (p1 + p2) / 2
        num = z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        n = num ** 2 / mde ** 2
    else:
        raise ValueError("kind must be 'mean' or 'proportion'")

    out = {"kind": kind, "mde": mde, "power": power, "alpha": alpha,
           "two_sided": two_sided, "rho": rho, "n_per_arm_raw": int(np.ceil(n))}

    deff = 1.0
    if icc is not None and cluster_size is not None:
        deff = cluster_design_effect(cluster_size, icc)
        n *= deff
        out["design_effect"] = deff
        out["cluster_size"] = cluster_size
        out["clusters_per_arm"] = int(np.ceil(n / cluster_size))

    out["n_per_arm"] = int(np.ceil(n))
    out["n_total"] = 2 * out["n_per_arm"]
    return out


def mde(kind, baseline, n, sd=None, power=0.80, alpha=0.05, two_sided=True, rho=0.0):
    """Smallest effect detectable at `n` per arm with the given power (inverse of
    sample_size). Means: closed form. Proportions: bisection on the absolute lift."""
    z_a, z_b = _zab(alpha, power, two_sided)
    if kind == "mean":
        if sd is None:
            raise ValueError("kind='mean' requires sd")
        sd_eff = _se_unit_mean(sd, rho)
        return float((z_a + z_b) * np.sqrt(2 * sd_eff ** 2 / n))
    if kind != "proportion":
        raise ValueError("kind must be 'mean' or 'proportion'")
    lo, hi = 1e-9, min(1 - baseline, baseline) - 1e-9
    for _ in range(100):
        mid = (lo + hi) / 2
        need = sample_size("proportion", baseline, mid, power=power, alpha=alpha,
                           two_sided=two_sided)["n_per_arm_raw"]
        if need > n:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def power(kind, baseline, mde, n, sd=None, alpha=0.05, two_sided=True, rho=0.0):
    """Power to detect `mde` at `n` per arm (the third leg of the quadrilateral)."""
    z_a, _ = _zab(alpha, 0.5, two_sided)  # power arg unused here
    if kind == "mean":
        if sd is None:
            raise ValueError("kind='mean' requires sd")
        se = _se_unit_mean(sd, rho) * np.sqrt(2 / n)
    elif kind == "proportion":
        p1, p2 = baseline, baseline + mde
        pbar = (p1 + p2) / 2
        se0 = np.sqrt(2 * pbar * (1 - pbar) / n)
        se1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
        # reject when |effect| > z_a*se0; power under the alternative (se1)
        return float(stats.norm.sf((z_a * se0 - abs(mde)) / se1)
                     + stats.norm.cdf((-z_a * se0 - abs(mde)) / se1))
    else:
        raise ValueError("kind must be 'mean' or 'proportion'")
    ncp = abs(mde) / se
    return float(stats.norm.sf(z_a - ncp) + stats.norm.cdf(-z_a - ncp))


def power_curve(kind, baseline, mde, ns, sd=None, alpha=0.05, two_sided=True, rho=0.0):
    """Power at each n in `ns` — the table behind a power curve."""
    return [{"n_per_arm": int(n),
             "power": power(kind, baseline, mde, n, sd=sd, alpha=alpha,
                            two_sided=two_sided, rho=rho)} for n in ns]


def duration(n_per_arm, daily_users_per_arm):
    """Calendar days to accrue the per-arm sample at a daily arrival rate."""
    return float(np.ceil(n_per_arm / daily_users_per_arm))


def format_design_report(res, daily_users_per_arm=None):
    """Render a sample_size() result as a plain-language sizing report."""
    L = "=" * 70
    out = [L, "EXPERIMENT DESIGN — SAMPLE SIZE", L,
           f"  metric kind        : {res['kind']}",
           f"  target MDE         : {res['mde']}",
           f"  power / alpha      : {res['power']:.0%} / {res['alpha']} "
           f"({'two' if res['two_sided'] else 'one'}-sided)"]
    if res.get("rho"):
        saved = res["rho"] ** 2
        out.append(f"  CUPED rho          : {res['rho']:+.2f}  "
                   f"(cuts required n by ~{saved:.0%})")
    out.append(f"  base n / arm       : {res['n_per_arm_raw']:,}")
    if "design_effect" in res:
        out += [f"  cluster size       : {res['cluster_size']}",
                f"  design effect      : x{res['design_effect']:.2f}  (ICC inflation)",
                f"  clusters / arm     : {res['clusters_per_arm']:,}"]
    out += [L,
            f"  -> n PER ARM       : {res['n_per_arm']:,}",
            f"  -> n TOTAL         : {res['n_total']:,}"]
    if daily_users_per_arm:
        d = duration(res["n_per_arm"], daily_users_per_arm)
        out.append(f"  -> DURATION        : {int(d)} days "
                   f"at {daily_users_per_arm:,}/arm/day")
    out.append(L)
    return "\n".join(out)
