# P1: Seed-based graph matching

## Setup
- Conda env `privacy_projects` from `student_projects/01_seed_based/environment.yml`.
- Extra package: matplotlib (figures only), `pip install matplotlib` inside the env.
- Raw data is read in place from `student_projects/01_seed_based/student/seed_based/`. Nothing is copied or modified.

## Run (from the `privacy` folder)
- Val metrics, tables and figures: `conda run -n privacy_projects python p1_seed_based/run.py val`
- Test predictions (18 instances x 6 seed configs = 108 files): `conda run -n privacy_projects python p1_seed_based/run.py test`
- Validity check of all prediction files: `conda run -n privacy_projects python p1_seed_based/run.py check`
- Optional `--jobs N` sets parallel workers (default 8). Each mode takes well under a minute on a laptop.

## Files
- `config.json`: frozen method settings (chosen on Train only, frozen before Val and Test).
- `src/data.py` loader, `src/features.py` structural features, `src/match.py` method, `src/baseline.py` seed-free baseline, `src/metrics.py` metrics, `src/check_predictions.py` checker, `src/figures.py` figures.
- `tune_util.py`, `tuning/`: Train-only tuning scripts (history of how `config.json` was chosen).
- `results/`: Val tables (csv) and figures (png), Test prediction log.
- `predictions/<seed_config>/<instance>/predictions.csv`: Test submission, columns exactly `node_a,node_b`, seed pairs included.
- `report.md`: method, settings, Val results, analysis, limitations.
