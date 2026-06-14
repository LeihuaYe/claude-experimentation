---
name: ab-cate
description: Estimate heterogeneous treatment effects (CATE) from an experiment and guard against subgroup-fishing false positives — regression-adjusted ATE (Lin's estimator), S/T/X-learner meta-learners for how the effect varies with covariates, honest (sample-split) estimation, and a Benjamini-Hochberg subgroup guard. Use when the user says "did it work better for some users", "estimate heterogeneous / conditional effects", "CATE", "uplift modeling", "which segment responds most", "is this subgroup effect real", "the effect was bigger for power users", "regression adjustment", or wants to slice an A/B result by segment without fooling themselves.
---

You are estimating how a treatment effect VARIES across users — and protecting the user from the single most common experiment-analysis mistake: slicing by segments until something looks significant. Pair with `ab-readout` (the overall verdict); this answers "for whom, and is that real?"

## Step 0 — The trap to name out loud
The average effect (ATE) hides the story, but **naive post-hoc subgroup analysis will lie**: test enough slices and one looks significant by chance, and Simpson's paradox can flip a sign on aggregation. So: estimate heterogeneity with a proper method, validate it honestly (out of sample), and correct for how many subgroups you tried.

## Step 1 — Inputs
- **arm** (2 groups), **outcome**, and the **covariates** that might modify the effect (pre-experiment features: tenure, prior activity, device, geo, baseline metric).
- Optional: a **subgroup column** the user wants to interrogate (e.g., country, plan tier).

## Step 2 — Compute with the library (don't hand-roll)
```python
from ab_cate import lin_estimator, t_learner, x_learner, honest_cate, subgroup_fishing_guard, cate_summary, format_cate_report

lin   = lin_estimator(X, T, Y)                  # regression-adjusted ATE (+ variance reduction vs naive)
model = x_learner(X, T, Y)                      # CATE(x); X-learner is the safe default, esp. imbalanced arms
summ  = cate_summary(model, X)                  # mean / p10 / p90 spread of the CATE

# "is the effect really different for these segments?" — fish honestly:
guard = subgroup_fishing_guard(T, Y, {"power_users": mask_a, "new_users": mask_b, ...})
print(format_cate_report(lin=lin, cate_summary=summ, guard=guard))
```
Choosing a learner: **X-learner** for imbalanced arms (default); **T-learner** when arms are balanced and ample; **S-learner** when the effect is small and you want one model. For a claim you'll act on, confirm with **`honest_cate`** (fit and estimate on different rows).

## Step 3 — Interpret honestly
- **Lead with the regression-adjusted ATE** (Lin's estimator): it's the unbiased headline AND lower-variance than a raw difference in means (report the variance reduction — it's the same free win as CUPED).
- **CATE spread**: if p10≈p90, the effect is roughly homogeneous — don't manufacture a segment story. Real spread → describe who responds, framed as a hypothesis.
- **Subgroup guard is the gate**: report `raw_significant` vs `bh_significant`. A subgroup that is raw-significant but does NOT survive BH is the fishing trap — say so explicitly. "It only worked for segment X" needs to survive the correction *and* ideally replicate out of sample.
- **Watch for Simpson's paradox**: if a subgroup story depends on aggregation, check segment balance across arms (and whether SRM is clean within the slice).

## Step 4 — Verdict
State: the adjusted ATE with CI, whether meaningful heterogeneity exists, which subgroups survive BH (if any), and a recommendation — ship overall / ship-and-monitor a real responder segment / "the heterogeneity is not real, don't carve it up." Never promote a fished subgroup to a decision without honest, corrected evidence.
