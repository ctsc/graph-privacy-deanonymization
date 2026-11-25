# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## State of the repo
- All four projects are complete (p1_seed_based, p2_seed_free, p3_membership_inference, p4_dp_reconstruction), each with run.py, frozen config.json, README.md, report.md, results/ and its submission outputs. Top level also has `PLAN.md`, `decisions.md`, `progress.md`, `projects.docx` (assignment text) and `student_projects/` (raw data and instructions).
- Read `PLAN.md` for build order, `decisions.md` for choices and open questions, `progress.md` for what is done. Update `progress.md` and `decisions.md` after each meaningful step.

## Layout
- `student_projects/`: raw data and `PROJECT_INSTRUCTIONS.txt` per project (P4 uses `DP_STUDENT_10STEPS.txt`). Read in place. Never copy, move or modify.
- STATUS: all projects done (see progress.md). Configs are frozen and Test outputs written; do not retune.
- Code goes in one folder per project: `p1_seed_based`, `p2_seed_free`, `p3_membership_inference`, `p4_dp_reconstruction`.
- Each project folder holds: `src/`, `run.py` (single entry point), `config.json` (frozen settings), `README.md` (run steps), `results/` (tables and figures), required outputs, `report.md`.
- P2 reuses P1 code (loader, metrics, features, Hungarian, rejection). P1 and P2 share graph data format but P2 must never read seed files.

## Environment
- Conda env `privacy_projects` is installed (pinned versions from `environment.yml`, plus matplotlib 3.11.2 via pip). All four `environment.yml` files are identical. No torch.
- Run a project from the `privacy` folder: `conda run -n privacy_projects python pN_name/run.py` (P1, P2, P4 take a mode: val / test / check).
- `conda run` rejects `python -c` scripts that contain newlines; for ad hoc snippets call `C:/Users/carte/miniforge3/envs/privacy_projects/python.exe` directly.

## Priority and order
- P1, then P3 (core pair). P2 then P4 are extra credit and must not start until the core pair is submittable.
- Ask the user before any decision that changes scope.

## Experiment rules (strict)
- Tune on Train/Val only. Freeze `config.json` before Test. Never tune on test or hidden data.
- Seeds 42, 43, 44 where required. Report every seed plus mean and sample std (ddof=1). No cherry-picking.
- Never invent results. Failures and gaps are stated in `report.md`.

### P1 seed-based
- 6 seed configs per instance: random/degree at 1/5/10%. 5% is the primary comparison.
- Each config uses only its own seed file. No transferring matches between configs. One-to-one only.
- Output `predictions.csv` per test instance per config, columns exactly `node_a,node_b`.
- Metrics: precision, recall, F1, coverage, unseeded recall, macro-averaged per instance (not pooled over nodes).

### P2 seed-free
- No seeds. Compare against a degree-only baseline. Report by family and difficulty. Same output format and metrics as P1 (no unseeded recall).

### P3 membership inference
- Pools after pixel dedupe: target train/val/test 100 each, shadow pool 1200 split 900 attack-train and 300 calibration. 3 dev shadows plus 1 calibration shadow, each 100 train / 100 val / 100 attack non-members.
- Fit scalers on each model's own train members only. Exclude target validation from attack test.
- Attacks (loss threshold and shadow logistic) for both fixed 600 epochs and early stopping (patience 10, restore best). Thresholds and attack models chosen on calibration shadow only, then frozen.
- TPR@5%FPR is the largest empirical TPR with FPR <= 0.05, no interpolation. Also report TPR/FPR at frozen threshold, ROC-AUC, train/test accuracy.

### P4 DP reconstruction
- Per step `g_t = (theta_t - theta_{t+1}) / 0.1`. Never divide the endpoint difference by 0.1.
- Three attacks: plain gradient matching, gradient matching with public class-mean/variance prior, public class-mean baseline. Observed thetas are fixed inputs to the objective.
- Use only `public.npz` for priors. Do not use the MIA Digits data. Never combine trials or scenarios, and never use truth inside the optimizer.
- Output NPZ per observation file with only key `images`, shape (trials, B, 64), finite, in [0, 1], trial and label order preserved.

## Style
- No em dashes anywhere, including code comments.
- Bullets over long paragraphs in READMEs and reports.

## Git
- No repo yet. When created, commits show only the user as author: no Co-Authored-By lines and no AI attribution (this overrides default trailers).
