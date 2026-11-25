# P3: Membership inference and early stopping

## Setup
- Conda env `privacy_projects` from `student_projects/03_membership_inference/environment.yml`, plus matplotlib (figures only).
- Data read in place: `student_projects/03_membership_inference/data/digits/digits.csv`. Never copied or modified.

## Run (from the `privacy` folder)
- `conda run -n privacy_projects python p3_membership_inference/run.py`
- Runs seeds 42, 43, 44 x {fixed 600 epochs, early stopping} in parallel. About 1 to 2 minutes on a laptop.

## Files
- `config.json`: frozen settings (pool sizes, MLP, early stopping, attack features, C grid, threshold rule).
- `src/data.py` pools, `src/mlp.py` training loop, `src/attack.py` features, attacks and metrics, `src/figures.py`.
- `results/per_seed.csv`: every seed, mode and attack (target and calibration-shadow metrics).
- `results/summary_mean_std.csv`: mean and sample std (ddof=1) over seeds.
- `results/shadow_models.csv`, `results/split_record_ids.json` (record ids of every pool, per seed), `results/roc_curves.json`, `results/learning_curves.json`.
- `results/roc_curves.png`, `results/learning_curves.png`.
- `report.md`: protocol, results, defense tradeoffs, limitations.
