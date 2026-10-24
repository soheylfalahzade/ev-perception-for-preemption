# Reproducibility

All reported numbers are produced by the commands below. Every scenario is generated from an explicit
integer seed; development seeds are `1000+` and reported seeds are `0..99`.

| result | command | seeds | approx. time |
|---|---|---|---|
| operating curves, map matching, latency (`docs/results/results.json`, `intervals.json`) | `evperc-eval --dev 40 --test 100 --out results/eval` | dev 1000-1039, test 0-99 | 25 min, 1 core |
| closed-loop preemption (`docs/results/sumo_summary.json`, `sumo_runs.csv`) | `evperc-sumo --seeds 12 --jobs 2 --out results/sumo` | 0-11 | 10 min, 2 cores |
| scenario animation (`docs/assets/scenario.gif`) | `evperc-gif --out docs/assets/scenario.gif --seed 11 --results docs/results/results.json` | 11 | < 1 min |

Environment: Python 3.12, package versions in `requirements-lock.txt`, SUMO 1.28.0 from the
`eclipse-sumo` wheel (`pip install -e ".[sumo]"`).

Determinism: the emulator and the evaluation use `numpy.random.default_rng(seed)`; re-running the
evaluation reproduces the reported figures exactly on the same package versions. Latency numbers
depend on the machine and are not deterministic. The SUMO study replays the same SUMO seed in every
condition (paired design); results vary with the SUMO version.

Checks: `make all` runs ruff, mypy and pytest; the CI workflow runs the same on Python 3.10-3.12.

Provenance of `docs/results/results.json`: written by `evperc-eval`; it includes the fitted fusion
weights, the operating thresholds chosen on development data, the full operating curves, and the
emulator configuration.
