# Graph Privacy: De-anonymization, Membership Inference, and DP Reconstruction

Practical projects on recovering private information from anonymized graphs,
model predictions, and differentially private model updates. Each project has
reproducible code, a frozen configuration, a full report, and verified outputs.

## Projects

| Folder | Project | Attack | Headline result |
|---|---|---|---|
| `p1_seed_based/` | P1: Seed-based graph de-anonymization | Match nodes between two anonymized graphs given a small seed set | Val macro F1 0.99 at 5% seeds (random or degree) |
| `p2_seed_free/` | P2: Seed-free graph de-anonymization | Same matching task with no seeds, via spectral matching (GRAMPA) plus refinement | Val macro F1 1.000 on all 9 instances |
| `p3_membership_inference/` | P3: Membership inference and early stopping | Loss-threshold and shadow-model attacks on an MLP, plus early stopping as a defense | Target AUC 0.83 (fixed) / 0.79 (early), logistic attack |
| `p4_dp_reconstruction/` | P4: Reconstruction from DP model updates | Recover training images from 10 consecutive clipped/noised updates | Exact recovery without noise; DP collapses attack to the class-mean template |

Each project folder contains:

- `run.py`: single entry point (`val` / `test` / `check` modes for P1, P2, P4; plain run for P3)
- `config.json`: frozen hyperparameters, chosen on Train/validation data only
- `src/`: implementation
- `results/`: tables and figures
- `report.md`: method, protocol, results, analysis, limitations
- Submission outputs: `predictions/` (P1, P2) or `outputs/` (P4)

## Setup

- Conda environment `privacy_projects` (Python with numpy, scipy, scikit-learn,
  pandas, matplotlib). Environment files are in `student_projects/*/environment.yml`.
- P2 reuses code from `p1_seed_based/src`; keep both folders side by side.

## Data

Course-provided data lives in `student_projects/` and is **not included in this
repository** (it is gitignored). Obtain it from the course materials and place
it in the same layout; all code reads it in place and never modifies it.

## Running

From the repo root, with the `privacy_projects` env active:

```bash
# P1: validation metrics, test predictions, output check
conda run -n privacy_projects python p1_seed_based/run.py val
conda run -n privacy_projects python p1_seed_based/run.py test
conda run -n privacy_projects python p1_seed_based/run.py check

# P2: same modes
conda run -n privacy_projects python p2_seed_free/run.py val
conda run -n privacy_projects python p2_seed_free/run.py test
conda run -n privacy_projects python p2_seed_free/run.py check

# P3: full experiment, seeds 42/43/44, both training modes (~1-2 min)
conda run -n privacy_projects python p3_membership_inference/run.py

# P4: sanity checks, validation, test outputs, output check
conda run -n privacy_projects python p4_dp_reconstruction/sanity.py
conda run -n privacy_projects python p4_dp_reconstruction/run.py val
conda run -n privacy_projects python p4_dp_reconstruction/run.py test
conda run -n privacy_projects python p4_dp_reconstruction/run.py check
```

## Methodology notes

- All hyperparameters were tuned on Train/validation splits only and frozen in
  `config.json` before Test predictions were written. Test truth is hidden.
- P3 reports seeds 42, 43, 44 individually plus mean and sample std (ddof=1).
- P4 uses only `public.npz` for priors; trials, scenarios, and seeds are never
  combined, and ground truth is never used inside the optimizer.

See `PLAN.md`, `decisions.md`, and `progress.md` for build order, design
choices, and verification history.
