# claude-experimentation — contributor notes

A small, dependency-light toolkit for trustworthy experiment analysis, usable as Claude Code skills or plain Python libraries. Skills: `ab-design` (pre-launch sizing), `ab-readout` (post-launch analysis), and `ab-cate` (heterogeneous effects).

## Layout
- `ab_design/` — pre-launch sizing. `design.py` is the quadrilateral (sample size ↔ MDE ↔ power ↔ alpha) for mean/proportion metrics, plus CUPED, cluster design effect, ratio-metric delta method, alpha allocation, switchback; `cli.py` / `__main__.py` is the `python -m ab_design` entrypoint. `SKILL.md` drops into `~/.claude/skills/ab-design/`.
- `ab_readout/` — post-launch analysis. `readout.py` is the pipeline (SRM → CUPED → effect+CI → Benjamini-Hochberg → verdict); `cli.py` / `__main__.py` is the `python -m ab_readout` entrypoint. `SKILL.md` drops into `~/.claude/skills/ab-readout/`.
- `ab_cate/` — heterogeneous effects. `cate.py` is Lin's regression-adjusted ATE, S/T/X-learner CATE, honest sample-split estimation, and a Benjamini-Hochberg subgroup-fishing guard; `cli.py` / `__main__.py` is the `python -m ab_cate` entrypoint. `SKILL.md` drops into `~/.claude/skills/ab-cate/`.
- `examples/make_synthetic.py` — synthetic data with a known ground truth; `example_output.txt` is the committed readout demo.
- `tests/test_design.py`, `tests/test_readout.py`, `tests/test_cate.py` — validate the statistics against ground truth (design against Monte-Carlo power; cate against known heterogeneity + FDR control).

## Conventions
- **Correctness first.** Every method is checked against a simulation with a known answer (`python tests/test_readout.py`). If you add a method, add a ground-truth test for it — plausible-looking stats are not enough.
- **Dependency-light.** numpy / scipy / pandas only; no heavy frameworks.
- Keep functions small and readable; prefer a few clear lines over premature abstraction.

## Roadmap
`ab-design`, `ab-readout`, and `ab-cate` ship today. Planned: **sequential / always-valid** inference and **causal inference without randomization** (DiD / synthetic control).

Contributions welcome — open an issue or PR.
