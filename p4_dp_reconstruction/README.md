# P4: Image reconstruction from ten consecutive (DP) parameter updates

## Setup
- Conda env `privacy_projects` from `student_projects/04_dp_reconstruction/environment.yml`, plus matplotlib (figures only).
- Inputs read in place from `student_projects/04_dp_reconstruction/` (`public.npz`, `validation/`, `observations/`). Only `public.npz` is used for priors. The MIA Digits data is never touched.

## Run (from the `privacy` folder)
- Sanity checks (gradient finite differences, noise level on validation): `conda run -n privacy_projects python p4_dp_reconstruction/sanity.py`
- Tuning grid on public validation: `conda run -n privacy_projects python p4_dp_reconstruction/tune.py`
- Validation evaluation, tables, figures: `conda run -n privacy_projects python p4_dp_reconstruction/run.py val`
- Test predictions with the frozen config: `conda run -n privacy_projects python p4_dp_reconstruction/run.py test`
- Output validity check: `conda run -n privacy_projects python p4_dp_reconstruction/run.py check`

## Files
- `config.json`: frozen attack settings and how each was chosen.
- `src/grad.py` per-step gradients and the matching objective with analytic derivatives, `src/attack.py` plain / prior / public-mean attacks, `src/evaluate.py` label-constrained Hungarian alignment, MSE, PSNR, `src/figures.py`.
- `results/`: `val_summary.csv`, `val_per_image.csv`, `val_utility_by_step.csv`, `tuning_validation.csv`, figures `val_mse_by_scenario.png`, `val_examples.png`, `val_convergence.png`.
- `outputs/seed{42,43,44}/batch{1,2,4}_eps_4.npz`: submission, key `images` only, shape (trials, B, 64), in [0, 1].
- `report.md`.
