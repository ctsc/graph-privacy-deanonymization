# P4 report: Reconstruction from ten consecutive DP updates

## Threat model and derivation
- Softmax regression, theta = W (10 x 64, row-major) then b (10). Every trial starts at the public model and takes 10 SGD steps (lr 0.1) on the same private batch.
- Per step: g_t = (theta_t - theta_{t+1}) / 0.1. The endpoint difference is never used as one gradient.
- Per-example gradient: r xt^T with r = softmax(W x + b) - e_y and xt = [x, 1]. Clipped to norm C = 1 (when clipping is on), averaged over the batch, plus N(0, sigma^2) per coordinate with DP.
- Checks (`sanity.py`, public validation originals):
  - Analytic gradient of the objective matches finite differences to 1e-8 or better, with and without clipping.
  - Residual between observed g_t and the true clipped mean gradient: exactly 0 for non-private and clip-only, and its std matches `sigma_per_step` in every DP file (for example batch1 eps 4: 8.188 vs 8.197; batch4 eps 4: 2.044 vs 2.049). So the noise model is as stated.

## Attacks
- Objective for one trial: sum_t 0.5 ||G_t(x) - g_t||^2, with each observed theta_t as a fixed input (not a simulated trajectory). x = sigmoid(u) keeps pixels in [0, 1]. Adam on u, best objective over restarts. Truth is never used inside the optimizer.
- **Plain:** random-normal init in logit space (3 restarts). For B = 1 with sigma = 0 the first init is exact: x = dW[k] / db[k] for the class row k with the largest |db[k]| (clipping scales both equally, so this also holds for clip-only).
- **Prior:** objective plus factor * sigma^2 * sum (x - mu_y)^2 / (2 var_y), with class means and variances from the 800 public images (variance floor 0.01). Inits: exact init when B = 1 and sigma = 0, the class mean, then random. With factor 1 this is the Gaussian MAP estimate.
- **Public mean:** the public class-mean image for each known label. Ignores all updates.
- Evaluation: label-constrained Hungarian alignment of slots within a trial, then per-image MSE on [0, 1] pixels, PSNR = 10 log10(1 / MSE).
- Each trial is attacked alone. No trial, scenario or seed shares information with another.

## Settings (chosen on public validation only, frozen in `config.json`)
- Adam lr 0.1, 1500 iterations, 3 restarts: best plain MSE on the sigma = 0 files (`results/tuning_validation.csv`). On DP files plain is bad for every setting (MSE 0.25 to 0.32).
- prior_factor 1.0 from {0.1, 0.3, 1, 3, 10}: mean DP-validation MSE 0.0544 / 0.0457 / **0.0446** / 0.0447 / 0.0447.
- The spec's prior weight factor * sigma^2 is 0 without noise. A small floor (1e-4, 1e-3) only made the sigma = 0 files worse, so it is 0.
- Submitted attack: the lowest eps 4 validation MSE among the three. Prior 0.044743 vs public mean 0.044744 vs plain 0.3048. Prior and mean are tied to 6 decimals; prior was chosen by the pre-set rule.

## Validation results (20 public images per file; mean per-image MSE, PSNR in dB)
Source: `results/val_summary.csv`. Ratio = MSE / public-mean MSE (below 1 beats the template).

| B | scenario | plain MSE | prior MSE | mean MSE | prior PSNR | prior ratio | prior beats mean (trials) |
|---|---|---|---|---|---|---|---|
| 1 | non-private | 0.0000 | 0.0000 | 0.0447 | 86.4 | 0.000 | 100% |
| 1 | clip only | 0.0000 | 0.0000 | 0.0447 | 86.4 | 0.000 | 100% |
| 1 | eps 16 | 0.3102 | 0.0445 | 0.0447 | 13.80 | 0.994 | 55% |
| 1 | eps 4 | 0.2920 | 0.0449 | 0.0447 | 13.76 | 1.004 | 40% |
| 1 | eps 1 | 0.2790 | 0.0448 | 0.0447 | 13.78 | 1.001 | 50% |
| 2 | non-private | 0.0321 | 0.0001 | 0.0447 | 43.2 | 0.002 | 100% |
| 2 | clip only | 0.0279 | 0.0001 | 0.0447 | 42.5 | 0.002 | 100% |
| 2 | eps 16 | 0.2959 | 0.0448 | 0.0447 | 13.76 | 1.001 | 60% |
| 2 | eps 4 | 0.3218 | 0.0447 | 0.0447 | 13.78 | 1.000 | 40% |
| 2 | eps 1 | 0.2864 | 0.0448 | 0.0447 | 13.78 | 1.001 | 30% |
| 4 | non-private | 0.0314 | 0.0003 | 0.0447 | 39.4 | 0.006 | 100% |
| 4 | clip only | 0.0283 | 0.0002 | 0.0447 | 39.2 | 0.005 | 100% |
| 4 | eps 16 | 0.2571 | 0.0441 | 0.0447 | 13.82 | 0.986 | 80% |
| 4 | eps 4 | 0.3006 | 0.0446 | 0.0447 | 13.80 | 0.997 | 80% |
| 4 | eps 1 | 0.2758 | 0.0447 | 0.0447 | 13.78 | 1.000 | 40% |

- Trials per file: 20 (B = 1), 10 (B = 2), 5 (B = 4). The trial fractions for B = 4 are out of 5 trials, so 80% is 4 of 5.
- Figures: `val_mse_by_scenario.png` (log scale), `val_examples.png` (trial 0 originals vs reconstructions), `val_convergence.png` (objective vs iteration).
- Utility of the released models (`val_utility_by_step.csv`, accuracy on the 200 public calibration images): 0.96 at step 0 everywhere. After 10 steps: non-private and clip-only 0.93 to 0.96; eps 16: 0.61 / 0.83 / 0.94 (B = 1 / 2 / 4); eps 4: 0.21 / 0.35 / 0.70; eps 1: 0.14 / 0.16 / 0.20.

## Analysis
- **Without noise the private images are recovered.** B = 1 is exact (PSNR 86 dB, float precision) from the gradient ratio, with or without clipping. B = 2 and 4 reach PSNR 39 to 43 dB with the prior attack. Clipping alone gives no protection: it rescales the gradient but keeps its direction.
- **Plain matching from random inits gets stuck for B > 1** (MSE about 0.03, better than the template but far from the prior attack's 0.0001 to 0.0003). With no noise the prior term is 0, so the gain comes only from starting at the class mean, which lands in the right basin. This shows the batch ambiguity problem: the averaged gradient has many approximate solutions.
- **With DP, plain matching is much worse than doing nothing** (MSE 0.26 to 0.32, 5.7 to 7.2 times the public-mean MSE). It fits the noise and produces salt-and-pepper images (see `val_examples.png`).
- **With DP, the prior attack collapses to the class template.** Its MSE ratio to the public mean is 0.986 to 1.004 at every epsilon and batch size, and it beats the template in 30% to 80% of trials, a coin flip. On the Test outputs the reconstructions are within a mean squared distance of 6e-5 to 1.5e-4 of the class template. A digit-shaped output here is the template, not identity recovery.
- **Why:** the clipped mean gradient has norm at most 1 spread over 650 coordinates, while the noise std per coordinate is 0.6 to 31 across validation scenarios (eps 4: 8.2 / 4.1 / 2.0 for B = 1 / 2 / 4). Even summed over 10 steps the signal is far below the noise. Larger B lowers sigma (sensitivity 2C/B) but mixes more images into one gradient; B = 4 at eps 16 shows the only hint of signal (ratio 0.986, 4 of 5 trials better).
- **Privacy and utility move together:** the same noise that stops reconstruction also wrecks the model (eps 4, B = 1: accuracy falls from 0.96 to 0.21 in 10 steps).

## Test submission
- `outputs/seed{42,43,44}/batch{1,2,4}_eps_4.npz`, 9 files, prior attack with the frozen config. `run.py check`: 9 / 9 valid (only key `images`, shape (trials, B, 64) = (100, 1, 64), (50, 2, 64), (25, 4, 64), finite, in [0, 1], trial and label-slot order preserved).
- No Test metrics: originals are hidden. The three seeds are repeats on the same 100 private images, so they are not independent datasets, and they were never combined.

## Limitations and failures
- Under eps 4 (the Test setting) the attack does not beat the public-mean baseline in a meaningful way on validation. This is a real failure to reconstruct, not proof that reconstruction is impossible: a finite optimization budget, one prior family and 3 restarts were tried.
- Plain matching for B > 1 without noise is a local-optimum failure; more restarts or a smarter init (for example per-class gradient factorization) would likely fix it.
- Validation has only 20 images per file and 5 trials for B = 4, so trial-level win rates are very noisy.
- Not tried: inferred labels, endpoint-only attack, stronger priors (for example a learned generative model on public data), changing-batch protocols.
