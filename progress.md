# progress.md

Status tracker. Update after each step. Decisions live in `decisions.md`.

## Overall
- [x] Read all instructions and projects.docx
- [x] PLAN.md, decisions.md written
- [x] CLAUDE.md, progress.md written
- [x] conda env privacy_projects installed (+ matplotlib)
- [x] User go-ahead given ("just go"). Q1, Q2 resolved with defaults (see decisions.md D5, D6)

## P1 seed-based (core)
- [x] Loader and metrics (p1_seed_based/src/data.py, metrics.py)
- [x] Structural features (src/features.py), seed-free structural baseline (src/baseline.py)
- [x] Seed method plus iterative expansion (src/match.py), works
- [x] Final Train tune (reject_tau, rounds, bandwidth), config.json frozen
- [x] Val results for 6 configs (run once): F1 random 1/5/10% 0.909/0.991/0.998, degree 0.924/0.987/0.996, baseline 0.074
- [x] Test predictions: 108 files, `run.py check` 108/108 valid
- [x] README, report.md, figures in results/

## P3 membership inference (core)
- [x] Splits and MLP training
- [x] Shadow models
- [x] Attacks, freeze
- [x] Seeds 42/43/44 results, figures
- [x] README, report.md
- Results (mean over seeds): AUC fixed loss 0.795, fixed logistic 0.833, early loss 0.775, early logistic 0.794. Test acc 0.867 fixed, 0.853 early.

## P2 seed-free (extra, after core pair)
- [x] Degree-only baseline, GRAMPA spectral + refinement + percolation (reuses P1 src), config frozen
- [x] Leakage check (truth not identity, relabeled B still F1 1.0)
- [x] Val: method F1 1.000 on all 9, degree baseline 0.052. Robustness probe: breaks at about 20% extra edge replacement (ER, SBM), PA survives to 20%
- [x] Test predictions: 18 files, `run.py check` 18/18 valid
- [x] README, report.md

## P4 DP reconstruction (extra, after P2)
- [x] sanity.py: analytic gradient matches finite differences, validation noise std equals sigma_per_step
- [x] plain, prior, public-mean attacks; tune.py on public validation only; config frozen
- [x] Validation: B1 non-private/clip exact (PSNR 86); B2/B4 no-noise prior PSNR 39 to 43; all DP scenarios prior is about equal to public mean (ratio 0.986 to 1.004), plain 5.7 to 7.2x worse
- [x] Test outputs: 9 NPZ, `run.py check` 9/9 valid
- [x] README, report.md

## Status
- ALL FOUR PROJECTS COMPLETE and submittable. Nothing pending except user review and git repo creation (user does it; no AI attribution in commits).

## Independent verification (2026-11-23, Devin CLI, read-only except reruns)
- Separate checker (outside the project folders) against the raw data: P1 108/108 files (header exactly `node_a,node_b`, one-to-one, ids exist, every seed pair of that config included); P2 18/18 valid and no P2 code references seed files; P3 pools 100/100/100 + 1,200 (900 dev, 300 calibration), disjoint, 297 unused, summary means and std (ddof=1) recomputed and equal; P4 9/9 NPZ with only `images`, shapes match the observation labels, finite, in [0, 1], gradients per step (`(states[:-1] - states[1:]) / lr`), priors from `public.npz` only. No em or en dashes.
- Reproducibility: reran P3 (`run.py`), P1 and P2 (`val`, `test`) with the frozen configs. P3 results and all P1/P2 predictions are byte-identical; P1/P2 Val tables differ only in wall-clock runtime. The original result files were restored so the runtimes quoted in the reports still match. Nothing was retuned.
- Human-only items are in `manual.md`.

## Validation numbers (final)
- P1 Val macro F1: random 1/5/10% 0.909 / 0.991 / 0.998, degree 0.924 / 0.987 / 0.996, seed-free structural baseline 0.074.
- P2 Val macro F1: 1.000 (degree baseline 0.052).
- P3 target AUC mean over seeds: fixed loss 0.795, fixed logistic 0.833, early loss 0.775, early logistic 0.794. TPR@5%FPR logistic 0.210 fixed, 0.177 early.
- P4 eps 4 validation MSE: prior 0.044743, public mean 0.044744, plain 0.3048.
