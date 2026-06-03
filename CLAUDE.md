# claude-experimentation — contributor notes

A small, dependency-light toolkit for trustworthy experiment analysis, usable as a Claude Code skill or a plain Python library. First skill: `ab-readout`.

## Layout
- `ab_readout/` — the library. `readout.py` is the pipeline (SRM → CUPED → effect+CI → Benjamini-Hochberg → verdict); `cli.py` / `__main__.py` is the `python -m ab_readout` entrypoint.
- `SKILL.md` — the Claude Code skill definition (drop into `~/.claude/skills/ab-readout/`).
- `examples/make_synthetic.py` — synthetic data with a known ground truth; `example_output.txt` is the committed demo.
- `tests/test_readout.py` — validates the statistics against that ground truth.

## Conventions
- **Correctness first.** Every method is checked against a simulation with a known answer (`python tests/test_readout.py`). If you add a method, add a ground-truth test for it — plausible-looking stats are not enough.
- **Dependency-light.** numpy / scipy / pandas only; no heavy frameworks.
- Keep functions small and readable; prefer a few clear lines over premature abstraction.

## Roadmap
`ab-readout` is the first skill. Planned: experiment **design** (power / MDE / switchback), **sequential / always-valid** inference, and **heterogeneous effects** (CATE).

Contributions welcome — open an issue or PR.
