# P2: Seed-free graph matching

## Setup
- Conda env `privacy_projects` from `student_projects/02_seed_free/environment.yml`, plus matplotlib (figures only).
- Data read in place from `student_projects/02_seed_free/student/seed_free/`. No seed file from any project is read.
- Reuses P1 code from `../p1_seed_based/src` (metrics, structural features, percolation matcher). Keep both folders side by side.

## Run (from the `privacy` folder)
- Val: `conda run -n privacy_projects python p2_seed_free/run.py val`
- Test predictions (18 files): `conda run -n privacy_projects python p2_seed_free/run.py test`
- Validity check: `conda run -n privacy_projects python p2_seed_free/run.py check`
- Robustness probe on Val (analysis only): `conda run -n privacy_projects python p2_seed_free/robustness.py`

## Files
- `config.json`: frozen settings.
- `src/data.py` loader (no seeds), `src/seedfree.py` method and degree-only baseline.
- `tune_util.py`: Train/Val evaluator used for the Train check and the leakage check.
- `results/`: Val tables and figures, robustness table and figure, Test prediction log.
- `predictions/<instance>/predictions.csv`: Test submission, columns exactly `node_a,node_b`.
- `report.md`.
