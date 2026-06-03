---
name: ab-readout
description: Read out / analyze an A/B test or online experiment trustworthily — SRM (sample ratio mismatch) check, CUPED variance reduction, effect size with confidence intervals, Benjamini-Hochberg multiple-testing correction across metrics, and a ship/no-ship verdict. Use when the user says "read out this experiment", "analyze these A/B results", "is this result significant", "did the test win", "check for SRM", "apply CUPED", "is my experiment valid", or shares experiment/treatment-control data and wants a decision.
---

You are running a trustworthy experiment readout. **A readout is a pipeline of gates, not a single p-value.** Treat the validity checks as blocking: a significant effect on an invalid experiment is worse than no result.

## Step 1 — Gather the inputs
From the user's data (a CSV path or a dataframe), identify:
- **arm column** — the assignment column (must have exactly 2 groups).
- **primary metric** — the one decision metric.
- **secondary metrics** and **guardrails** (metrics that must NOT regress, e.g. latency, error rate, tickets).
- **binary metrics** — which metrics are 0/1 (e.g. conversion).
- **covariate** (optional) — a *pre-experiment* column correlated with the primary metric, for CUPED. (Must be pre-treatment, or it biases the effect.)

Ask only for what you can't infer from column names.

## Step 2 — Run the pipeline
Use the bundled `ab_readout` library (do not hand-roll the stats):

```python
from ab_readout import run_readout, format_report
res = run_readout(df, arm_col=..., primary=..., metrics=[...],
                  binary=[...], guardrails=[...], covariate=...)
print(format_report(res))
```

## Step 3 — Interpret conservatively
- **If SRM fails (p ≤ 0.001): STOP.** Do not report effects — the randomization is broken; surface it as the headline and recommend investigating assignment/logging. Nothing downstream is trustworthy.
- **Report CUPED's benefit** when a covariate is given (variance reduction %, CI narrowing). If the covariate is weakly correlated (|corr| < ~0.2), say CUPED won't help much.
- **Judge significance on the BH-corrected result**, not raw p-values. Explicitly call out any metric that is raw-significant but does NOT survive BH — that's the multiple-comparisons trap.
- **Guardrails are veto power.** A primary win with a regressed guardrail is not a ship.
- Report the **effect size and CI**, not just significance. "We can't rule out zero" and "we're confident it's tiny" are different conclusions — distinguish them.

## Step 4 — Give the verdict
State plainly: SRM status, primary effect (CUPED-adjusted) with CI, what survived BH, guardrail status, and a ship / don't-ship / inconclusive recommendation with the reason. Do not overclaim — if the CI is wide, say the test is underpowered rather than "no effect".
