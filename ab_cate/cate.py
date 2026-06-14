"""Beyond the average effect: heterogeneous treatment effects, done honestly.

The ATE hides the story, and naive subgroup analysis will lie to you. This module
estimates how the effect varies with covariates (CATE) and guards against the
multiple-testing trap that makes fished subgroups look real:

  - lin_estimator        — regression adjustment (Lin's estimator): the ATE with
                           treatment x centered-covariate interactions. Unbiased, and
                           variance-reduced vs a raw difference in means (CUPED's cousin).
  - s/t/x_learner        — meta-learners for CATE(x). T-learner fits a model per arm;
                           S-learner one augmented model; X-learner imputes effects and
                           is the robust choice under imbalanced arms.
  - honest_cate          — fit on one split, estimate on another. The discipline that
                           stops you from discovering and confirming heterogeneity on the
                           same rows (which is how subgroup analysis fools people).
  - subgroup_fishing_guard — test each proposed subgroup for an effect DIFFERENT from the
                           rest, then Benjamini-Hochberg across all the subgroups you tried.
                           Fish 20 slices and the guard demotes the ones that are noise.

Dependency-light: numpy + scipy only. Base learner is a (ridge) linear model, which
captures linear heterogeneity and keeps every method checkable against known ground truth.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _2d(X):
    X = np.asarray(X, float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _fit_linear(X, y, ridge=0.0):
    """Ridge/OLS with an unpenalized intercept; returns coefficients [b0, b1..]."""
    X = _2d(X)
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    p = A.shape[1]
    R = ridge * np.eye(p)
    R[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + R, A.T @ np.asarray(y, float))


def _predict(coef, X):
    X = _2d(X)
    return np.hstack([np.ones((X.shape[0], 1)), X]) @ coef


def _augment(X, t):
    """[X, t, X*t] — lets a linear base express effect heterogeneity (the S-learner)."""
    X = _2d(X)
    t = np.asarray(t, float).reshape(-1, 1)
    return np.hstack([X, t, X * t])


def lin_estimator(X, T, Y):
    """Lin's regression-adjusted ATE: regress Y on [T, X_centered, T*X_centered].

    The coefficient on T is the ATE — unbiased under randomization, with variance
    reduced by the covariates (the regression-adjustment generalization of CUPED).
    Returns the ATE with CI/p AND the unadjusted difference in means for comparison.
    """
    X = _2d(X)
    T = np.asarray(T, float)
    Y = np.asarray(Y, float)
    Xc = X - X.mean(0)
    A = np.hstack([np.ones((len(Y), 1)), T[:, None], Xc, T[:, None] * Xc])
    n, p = A.shape
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ beta
    sigma2 = resid @ resid / (n - p)
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se = float(np.sqrt(cov[1, 1]))
    ate = float(beta[1])
    tcrit = stats.t.ppf(0.975, n - p)
    p_val = float(2 * stats.t.sf(abs(ate / se), n - p))

    c, t = Y[T == 0], Y[T == 1]
    dm = float(t.mean() - c.mean())
    dm_se = float(np.sqrt(c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t)))
    return {"ate": ate, "se": se, "ci": (ate - tcrit * se, ate + tcrit * se), "p": p_val,
            "naive_diff_means": dm, "naive_se": dm_se,
            "variance_reduction": float(1 - (se / dm_se) ** 2) if dm_se else 0.0}


def s_learner(X, T, Y, ridge=0.0):
    """One model on [X, T, X*T]; CATE(x) = f(x, 1) - f(x, 0)."""
    X = _2d(X)
    T = np.asarray(T, float)
    coef = _fit_linear(_augment(X, T), Y, ridge)

    def cate(Xn):
        Xn = _2d(Xn)
        one = np.ones(len(Xn))
        return _predict(coef, _augment(Xn, one)) - _predict(coef, _augment(Xn, one * 0))
    return {"cate": cate, "ate": float(cate(X).mean()), "kind": "S-learner"}


def t_learner(X, T, Y, ridge=0.0):
    """Two models, one per arm; CATE(x) = m1(x) - m0(x)."""
    X, T, Y = _2d(X), np.asarray(T), np.asarray(Y, float)
    m0 = _fit_linear(X[T == 0], Y[T == 0], ridge)
    m1 = _fit_linear(X[T == 1], Y[T == 1], ridge)

    def cate(Xn):
        return _predict(m1, Xn) - _predict(m0, Xn)
    return {"cate": cate, "ate": float(cate(X).mean()), "kind": "T-learner"}


def x_learner(X, T, Y, ridge=0.0, propensity=None):
    """X-learner — robust under imbalanced arms.

    1) m0, m1 per arm. 2) impute effects: treated D1=Y-m0(X), control D0=m1(X)-Y.
    3) model tau1, tau0 on the imputed effects. 4) CATE = e*tau0 + (1-e)*tau1.
    """
    X, T, Y = _2d(X), np.asarray(T), np.asarray(Y, float)
    c, t = T == 0, T == 1
    m0 = _fit_linear(X[c], Y[c], ridge)
    m1 = _fit_linear(X[t], Y[t], ridge)
    d1 = Y[t] - _predict(m0, X[t])
    d0 = _predict(m1, X[c]) - Y[c]
    tau1 = _fit_linear(X[t], d1, ridge)
    tau0 = _fit_linear(X[c], d0, ridge)
    e = float(t.mean()) if propensity is None else propensity

    def cate(Xn):
        return e * _predict(tau0, Xn) + (1 - e) * _predict(tau1, Xn)
    return {"cate": cate, "ate": float(cate(X).mean()), "kind": "X-learner",
            "propensity": e}


_LEARNERS = {"s": s_learner, "t": t_learner, "x": x_learner}


def honest_cate(X, T, Y, learner="x", ridge=0.0, test_frac=0.5, seed=0):
    """Fit the meta-learner on a training split, estimate CATE on a held-out split.

    Discovering and confirming heterogeneity on the same rows is how subgroup analysis
    lies; honest estimation separates the two. Returns CATE on the holdout + the indices.
    """
    X, T, Y = _2d(X), np.asarray(T), np.asarray(Y, float)
    n = len(Y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(round(n * (1 - test_frac)))
    tr, te = idx[:cut], idx[cut:]
    fit = _LEARNERS[learner](X[tr], T[tr], Y[tr], ridge=ridge)
    cate_te = fit["cate"](X[te])
    return {"cate": cate_te, "test_idx": te, "train_idx": tr,
            "ate_holdout": float(cate_te.mean()), "kind": fit["kind"] + " (honest)"}


def _bh(pvalues, alpha=0.05):
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * alpha
    passing = p[order] <= thresh
    kmax = int(np.where(passing)[0].max() + 1) if passing.any() else 0
    disc = np.zeros(m, bool)
    disc[order[:kmax]] = True
    return disc


def _effect(T, Y, mask):
    a, b = Y[mask & (T == 0)], Y[mask & (T == 1)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    return b.mean() - a.mean(), a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)


def subgroup_fishing_guard(T, Y, subgroups, alpha=0.05):
    """For each proposed subgroup, test whether its treatment effect DIFFERS from the
    rest of the sample (an interaction test), then Benjamini-Hochberg across every
    subgroup you tested. Fish enough slices and raw p<alpha flags some by luck; the BH
    correction is what tells you which 'the effect is bigger for X' claims are real.
    """
    T = np.asarray(T)
    Y = np.asarray(Y, float)
    rows, pvals = [], []
    for name, mask in subgroups.items():
        mask = np.asarray(mask, bool)
        eff_in, var_in = _effect(T, Y, mask)
        eff_out, var_out = _effect(T, Y, ~mask)
        if np.isnan(eff_in) or np.isnan(eff_out):
            continue
        se = np.sqrt(var_in + var_out)
        delta = eff_in - eff_out
        p = float(2 * stats.norm.sf(abs(delta / se))) if se > 0 else 1.0
        rows.append({"subgroup": name, "n": int(mask.sum()), "effect": float(eff_in),
                     "effect_rest": float(eff_out), "delta_vs_rest": float(delta), "p": p})
        pvals.append(p)
    disc = _bh(pvals, alpha)
    for r, d in zip(rows, disc):
        r["bh_real"] = bool(d)
    return {"subgroups": rows, "n_tested": len(rows),
            "raw_significant": int(sum(r["p"] < alpha for r in rows)),
            "bh_significant": int(disc.sum()), "alpha": alpha}


def format_cate_report(lin=None, cate_summary=None, guard=None):
    """Render any combination of the ATE, a CATE summary, and the subgroup guard."""
    L = "=" * 70
    out = []
    if lin:
        out += [L, "ATE — REGRESSION ADJUSTED (Lin's estimator)", L,
                f"  ATE                : {lin['ate']:+.4f}  95% CI "
                f"[{lin['ci'][0]:+.4f}, {lin['ci'][1]:+.4f}]  p={lin['p']:.4f}",
                f"  naive diff-in-means: {lin['naive_diff_means']:+.4f}  (se {lin['naive_se']:.4f})",
                f"  variance reduction : {lin['variance_reduction']:.1%}  "
                f"(adjusted se {lin['se']:.4f})"]
    if cate_summary:
        s = cate_summary
        out += ["", L, f"CATE — {s['kind']}", L,
                f"  mean (≈ATE)        : {s['mean']:+.4f}",
                f"  spread (p10..p90)  : {s['p10']:+.4f} .. {s['p90']:+.4f}",
                f"  -> {'heterogeneity present' if s['p90'] - s['p10'] > 1e-6 else 'roughly homogeneous'}"]
    if guard:
        out += ["", L, "SUBGROUP FISHING GUARD (effect vs the rest, BH-corrected)", L,
                f"  subgroups tested   : {guard['n_tested']}",
                f"  raw p<{guard['alpha']}        : {guard['raw_significant']}",
                f"  REAL after BH      : {guard['bh_significant']}"]
        for r in sorted(guard["subgroups"], key=lambda x: x["p"]):
            out.append(f"    {r['subgroup']:18s} n={r['n']:>7,}  Δvs-rest={r['delta_vs_rest']:+.4f}"
                       f"  p={r['p']:.4f}" + ("  <-- real" if r["bh_real"] else "  (noise)"))
    out.append(L)
    return "\n".join(out)


def cate_summary(model, X):
    """Quantile summary of CATE(X) for a fitted learner dict."""
    c = model["cate"](_2d(X))
    return {"kind": model["kind"], "mean": float(c.mean()),
            "p10": float(np.percentile(c, 10)), "p90": float(np.percentile(c, 90))}
