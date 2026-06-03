"""A/B test readout: SRM -> CUPED -> effect+CI -> Benjamini-Hochberg -> verdict.

A trustworthy readout is a pipeline of gates, not a single p-value:
  1. SRM     — is the randomization even valid? (if not, nothing else matters)
  2. CUPED   — free variance reduction from a pre-experiment covariate
  3. effect  — difference in means / proportions with a proper CI
  4. BH-FDR  — don't get fooled by the best of several borderline metrics
  5. verdict — a ship / no-ship call that respects all of the above
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def srm_check(arm, expected_ratio=0.5, alpha=0.001):
    """Chi-square goodness-of-fit on the assignment split. alpha is small by
    design — SRM is a data-quality alarm, you want few false alarms."""
    counts = arm.value_counts().sort_index()
    if len(counts) != 2:
        raise ValueError(f"expected exactly 2 arms, got {list(counts.index)}")
    obs = counts.values.astype(float)
    n = obs.sum()
    exp = np.array([expected_ratio, 1 - expected_ratio]) * n
    chi2 = ((obs - exp) ** 2 / exp).sum()
    p = float(stats.chi2.sf(chi2, df=1))
    return {"labels": list(counts.index), "counts": obs.tolist(),
            "chi2": float(chi2), "p": p, "alpha": alpha, "passed": p > alpha}


def cuped(y, x):
    """Return CUPED-adjusted y, theta, and realized variance reduction.
    Y_adj = Y - theta*(X - E[X]),  theta = Cov(Y,X)/Var(X), fit on pooled data."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    var_red = 1 - np.var(y_adj, ddof=1) / np.var(y, ddof=1)
    return y_adj, float(theta), float(var_red)


def mean_effect(control, treatment):
    """Welch (unequal-variance) two-sample difference in means + 95% CI + p."""
    a, b = np.asarray(control, float), np.asarray(treatment, float)
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    diff = b.mean() - a.mean()
    se = np.sqrt(va / na + vb / nb)
    dof = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    tcrit = stats.t.ppf(0.975, dof)
    p = float(2 * stats.t.sf(abs(diff / se), dof))
    return {"effect": float(diff), "se": float(se),
            "ci": (float(diff - tcrit * se), float(diff + tcrit * se)),
            "p": p, "control_mean": float(a.mean())}


def proportion_effect(control, treatment):
    """Two-proportion z-test for binary metrics (0/1)."""
    a, b = np.asarray(control, float), np.asarray(treatment, float)
    p1, p2, n1, n2 = a.mean(), b.mean(), len(a), len(b)
    pooled = (a.sum() + b.sum()) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    diff = p2 - p1
    p = float(2 * stats.norm.sf(abs(diff / se))) if se > 0 else 1.0
    return {"effect": float(diff), "se": float(se),
            "ci": (float(diff - 1.96 * se), float(diff + 1.96 * se)),
            "p": p, "control_mean": float(p1)}


def benjamini_hochberg(pvalues, alpha=0.05):
    """Return (discovery_mask, per-metric BH threshold) in input order."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * alpha
    passing = p[order] <= thresh
    kmax = int(np.where(passing)[0].max() + 1) if passing.any() else 0
    disc = np.zeros(m, bool)
    disc[order[:kmax]] = True
    crit = np.empty(m)
    crit[order] = thresh
    return disc, crit


def run_readout(df, arm_col, primary, metrics=None, binary=None, guardrails=None,
                covariate=None, srm_alpha=0.001, fdr_alpha=0.05, expected_ratio=0.5):
    """Full readout. Returns a structured dict; pass to format_report() for text."""
    binary = set(binary or [])
    guardrails = set(guardrails or [])
    metrics = list(metrics or [primary])
    if primary not in metrics:
        metrics = [primary] + metrics

    arm = df[arm_col]
    labels = sorted(arm.unique())
    if len(labels) != 2:
        raise ValueError(f"expected exactly 2 arms in '{arm_col}', got {labels}")
    control_label, treat_label = labels

    srm = srm_check(arm, expected_ratio, srm_alpha)

    cuped_info = None
    df = df.copy()
    if covariate and primary not in binary:
        y_adj, theta, var_red = cuped(df[primary], df[covariate])
        df["__primary_cuped"] = y_adj
        cuped_info = {"covariate": covariate, "theta": theta,
                      "variance_reduction": var_red,
                      "corr": float(np.corrcoef(df[primary], df[covariate])[0, 1])}

    c, t = df[arm == control_label], df[arm == treat_label]
    results = []
    for m in metrics:
        if m in binary:
            eff = proportion_effect(c[m], t[m])
        elif m == primary and cuped_info:
            eff = mean_effect(c["__primary_cuped"], t["__primary_cuped"])
            eff["raw"] = mean_effect(c[m], t[m])
        else:
            eff = mean_effect(c[m], t[m])
        role = "primary" if m == primary else "guardrail" if m in guardrails else "secondary"
        results.append({"metric": m, "role": role, **eff})

    disc, crit = benjamini_hochberg([r["p"] for r in results], fdr_alpha)
    for r, d, q in zip(results, disc, crit):
        r["bh_discovery"], r["bh_threshold"] = bool(d), float(q)

    return {"srm": srm, "cuped": cuped_info, "metrics": results, "fdr_alpha": fdr_alpha,
            "control_label": control_label, "treat_label": treat_label,
            "n_control": len(c), "n_treat": len(t)}


def _rel(e):
    return e["effect"] / e["control_mean"] if e["control_mean"] else float("nan")


def format_report(res):
    """Render the structured readout as a plain-language report."""
    L = "=" * 70
    out = []
    s = res["srm"]
    out += [L, "[1] SAMPLE RATIO MISMATCH (randomization sanity gate)", L,
            f"  {res['control_label']}={res['n_control']:,}   {res['treat_label']}={res['n_treat']:,}",
            f"  chi2={s['chi2']:.3f}  p={s['p']:.4f}  (alpha={s['alpha']})  ->  "
            + ("PASS — randomization looks valid" if s["passed"]
               else "FAIL — STOP, results are not trustworthy")]

    if res["cuped"]:
        ci = res["cuped"]
        narrower = 1 - np.sqrt(max(0.0, 1 - ci["variance_reduction"]))
        out += ["", L, "[2] CUPED VARIANCE REDUCTION", L,
                f"  covariate={ci['covariate']}  corr={ci['corr']:+.3f}  "
                f"variance reduction={ci['variance_reduction']:.1%}  "
                f"(=> CI ~{narrower:.0%} narrower, same users)"]

    prim = next(r for r in res["metrics"] if r["role"] == "primary")
    fmt = lambda e: (f"abs={e['effect']:+.4f}  rel={_rel(e):+.2%}  "
                     f"95% CI [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}]  p={e['p']:.4f}")
    out += ["", L, "[3] PRIMARY EFFECT", L]
    if "raw" in prim:
        out += [f"  raw    {fmt(prim['raw'])}", f"  CUPED  {fmt(prim)}"]
    else:
        out += [f"  {fmt(prim)}"]

    out += ["", L, f"[4] METRIC PANEL + BENJAMINI-HOCHBERG FDR (alpha={res['fdr_alpha']})", L]
    for r in sorted(res["metrics"], key=lambda x: x["p"]):
        out.append(f"  {r['metric']:18s}[{r['role']:9s}] effect={r['effect']:+.4f}  "
                   f"rel={_rel(r):+.2%}  p={r['p']:.4f}  q<={r['bh_threshold']:.4f}"
                   + ("  <-- discovery" if r["bh_discovery"] else ""))

    disc = [r["metric"] for r in res["metrics"] if r["bh_discovery"]]
    gbad = [r["metric"] for r in res["metrics"] if r["role"] == "guardrail" and r["bh_discovery"]]
    ship = s["passed"] and prim["bh_discovery"] and not gbad
    out += ["", L, "[5] VERDICT", L,
            f"  - SRM: {'PASS' if s['passed'] else 'FAIL'}",
            f"  - Significant after BH correction: {', '.join(disc) if disc else 'none'}",
            f"  - Guardrails: {'clean' if not gbad else 'REGRESSED -> ' + ', '.join(gbad)}",
            f"  - RECOMMENDATION: {'SHIP' if ship else 'DO NOT SHIP'}"]
    return "\n".join(out)
