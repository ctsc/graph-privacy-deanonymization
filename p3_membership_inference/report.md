# P3 report: Membership inference and early stopping

## Data and protocol
- Digits CSV, 1797 records, 64 pixels. The class column is named `income` in the supplied file; it holds the digit labels 0 to 9 (counts 174 to 183 per class).
- Pixel dedupe: 0 exact duplicate rows found, so no records were dropped. The step is still in the code.
- Per seed, disjoint stratified pools: target train 100, target val 100, target test 100, shadow pool 1200 (297 records unused).
  - Shadow pool split into 900 attack-training and 300 calibration records.
  - 3 dev shadows from the 900: each 100 train / 100 val / 100 attack nonmembers, disjoint.
  - 1 calibration shadow from the 300: 100 train / 100 val / 100 nonmembers.
  - Disjointness is asserted in code. Record ids per pool are in `results/split_record_ids.json`.
- The seed (42, 43, 44) controls the pools, the MLP initialization and the batch order, so each seed is a full independent repeat.
- Preprocessing: `StandardScaler` fit on each model's own 100 training members only.

## Models
- sklearn `MLPClassifier`, hidden (64, 32), ReLU, Adam lr 0.001, L2 alpha 1e-4, batch size 32, trained with a manual `partial_fit` epoch loop.
- Fixed: 600 epochs.
- Early stopping: monitor the model's own validation loss every epoch, patience 10, restore the best-epoch weights.
- Shadows use the same mode as the target they attack (fixed shadows for the fixed target, early-stopped shadows for the early target).
- Batch size 32 was set in `config.json` before any run (so an epoch is several Adam steps, not one full-batch step). It was not tuned.

## Attacks
- Loss threshold: score = negative cross-entropy of the true label. Nothing is learned from the dev shadows.
- Shadow logistic: features true-label probability, loss, entropy, max probability, correctness, task-label one-hot (15 features). Standardized logistic regression trained on 600 dev shadow records (3 shadows x 100 members + 100 nonmembers).
- Selection on the calibration shadow only, then frozen before the target is attacked:
  - logistic C from {0.01, 0.1, 1, 10} by calibration ROC-AUC,
  - for both attacks, the threshold with the largest calibration TPR subject to FPR <= 0.05.
- Target attack-Test: 100 target-train members vs 100 target-test nonmembers. Target validation is excluded.
- TPR@5%FPR: largest empirical TPR over thresholds with FPR <= 0.05, no interpolation (at most 5 of 100 nonmembers flagged).

## Results per seed (target attack-Test)
Source: `results/per_seed.csv`.

| seed | mode | attack | epochs | train acc | test acc | AUC | TPR@5%FPR | TPR at frozen thr | FPR at frozen thr | calib AUC | C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | fixed | loss | 600 | 1.00 | 0.93 | 0.798 | 0.08 | 0.03 | 0.03 | 0.752 | |
| 42 | fixed | logistic | 600 | 1.00 | 0.93 | 0.810 | 0.15 | 0.20 | 0.07 | 0.823 | 10 |
| 42 | early | loss | 119 | 1.00 | 0.91 | 0.762 | 0.08 | 0.00 | 0.00 | 0.749 | |
| 42 | early | logistic | 119 | 1.00 | 0.91 | 0.768 | 0.09 | 0.00 | 0.00 | 0.815 | 10 |
| 43 | fixed | loss | 600 | 1.00 | 0.85 | 0.775 | 0.02 | 0.08 | 0.07 | 0.724 | |
| 43 | fixed | logistic | 600 | 1.00 | 0.85 | 0.834 | 0.32 | 0.13 | 0.03 | 0.788 | 10 |
| 43 | early | loss | 136 | 1.00 | 0.83 | 0.758 | 0.02 | 0.06 | 0.07 | 0.719 | |
| 43 | early | logistic | 136 | 1.00 | 0.83 | 0.813 | 0.31 | 0.10 | 0.02 | 0.778 | 0.01 |
| 44 | fixed | loss | 600 | 1.00 | 0.82 | 0.812 | 0.06 | 0.09 | 0.08 | 0.769 | |
| 44 | fixed | logistic | 600 | 1.00 | 0.82 | 0.854 | 0.16 | 0.20 | 0.08 | 0.812 | 10 |
| 44 | early | loss | 273 | 1.00 | 0.82 | 0.805 | 0.09 | 0.10 | 0.08 | 0.756 | |
| 44 | early | logistic | 273 | 1.00 | 0.82 | 0.802 | 0.13 | 0.21 | 0.10 | 0.768 | 1 |

- "epochs" for early stopping is the restored best epoch.

## Mean and sample std over seeds (ddof = 1)
Source: `results/summary_mean_std.csv`.

| mode | attack | test acc | AUC | TPR@5%FPR | TPR at frozen thr | FPR at frozen thr |
|---|---|---|---|---|---|---|
| fixed | loss | 0.867 +/- 0.057 | 0.795 +/- 0.019 | 0.053 +/- 0.031 | 0.067 +/- 0.032 | 0.060 +/- 0.026 |
| fixed | logistic | 0.867 +/- 0.057 | 0.833 +/- 0.022 | 0.210 +/- 0.095 | 0.177 +/- 0.040 | 0.060 +/- 0.026 |
| early | loss | 0.853 +/- 0.049 | 0.775 +/- 0.026 | 0.063 +/- 0.038 | 0.053 +/- 0.050 | 0.050 +/- 0.044 |
| early | logistic | 0.853 +/- 0.049 | 0.794 +/- 0.023 | 0.177 +/- 0.117 | 0.103 +/- 0.105 | 0.040 +/- 0.053 |

- Train accuracy is 1.00 for every target model in both modes. Early-stopped targets stopped at best epoch 119 / 136 / 273 (mean 176 +/- 84).
- Calibration shadow AUC (mean over seeds): fixed loss 0.748, fixed logistic 0.808, early loss 0.741, early logistic 0.787. These are close to the target values, so the shadows are a fair stand-in.

## Analysis
- **The attacks work, clearly above chance.** Every target AUC is 0.76 to 0.85 (0.5 is random). The cause is visible in `results/learning_curves.png`: with 100 training images the MLP reaches 100% train accuracy within about 40 epochs, then train loss keeps falling toward 1e-3 or lower while val loss stays near 0.3 to 0.7. Members get much lower loss than nonmembers.
- **The shadow logistic attack beats the plain loss threshold** in AUC in 5 of 6 seed-mode pairs (mean 0.833 vs 0.795 fixed, 0.794 vs 0.775 early) and is much stronger at low FPR (TPR@5%FPR 0.21 vs 0.05 fixed). The extra features (entropy, max probability, label one-hot) help separate members from nonmembers that are confidently correct anyway.
- **Low-FPR numbers are noisy.** With only 100 nonmembers, FPR <= 5% means at most 5 false alarms, so TPR@5%FPR moves a lot between seeds (logistic: 0.15, 0.32, 0.16 fixed). The frozen thresholds land at FPR 0.00 to 0.10 on the target, not exactly 0.05, which is expected when moving a threshold from one 200-record shadow set to another.
- **Early stopping as a defense: small gain, small cost.**
  - AUC drops by 0.020 (loss) and 0.039 (logistic) on average. The logistic AUC drop happens in all 3 seeds; the loss AUC drop in all 3 seeds too but is within about one std.
  - TPR@5%FPR barely changes (loss 0.053 to 0.063, logistic 0.210 to 0.177). The instructions note early stopping need not improve every metric, and here it does not help at low FPR in a meaningful way.
  - Test accuracy falls slightly, 0.867 to 0.853 (two fewer correct out of 100 in seeds 42 and 43, unchanged in seed 44).
  - Why the effect is small: patience 10 on a flat val loss lets training run 119 to 273 epochs, long after train accuracy hits 100%. The early models are still strongly overfit (train loss about 1e-2 to 1e-3 at the stop point).
- **Tradeoff summary:** early stopping here is close to free in utility and gives a modest AUC reduction, but it is not a strong defense. Stronger regularization (larger alpha, smaller model, fewer epochs, DP training) would be needed to push AUC toward 0.5.

## Limitations
- Three seeds and 100 + 100 attack-Test records per seed: confidence intervals are wide, especially for TPR@5%FPR.
- Only one architecture and one regularization setting were tested. No per-class attack, no label-only attack.
- C selection on calibration AUC picked C = 10 in 4 of 6 runs (the largest in the grid). A wider grid was not tried.
- The calibration shadow is used for both C selection and threshold selection, so its own AUC is slightly optimistic. It is never used for target reporting.
