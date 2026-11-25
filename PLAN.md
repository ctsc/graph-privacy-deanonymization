# PLAN

Goal: finish all 4 projects in ~7 days, simple methods, honest reporting.
Order: P1, P3 (core pair), then P2, P4 (extra). No extra project starts until the core pair is submittable.

## Ground rules (apply everywhere)
- Raw data read in place from `student_projects/`. Never copied or modified.
- Tune on Train/Val only. Freeze config before Test. Never touch hidden data.
- Seeds 42, 43, 44 where required. Report each seed, mean, sample std (ddof=1).
- No em dashes anywhere. Bullets in READMEs and reports.
- Never invent numbers. Failures and gaps go in report.md.
- Git: user creates repo later. Commits authored by user only, no Co-Authored-By, no AI attribution (overrides the default trailer).

## Environment
- Conda env `privacy_projects` installed (pinned versions) plus matplotlib. No torch.
- Plan: numpy/scipy/sklearn/pandas/networkx(optional)/matplotlib only. Run via `conda run -n privacy_projects python ...`.

## Folder layout (one per project)
```
privacy/pN_name/
  src/            code
  config.json     frozen settings
  run.py          single entry point (run script)
  README.md       run steps
  results/        tables (csv), figures (png)
  predictions/ or outputs/   required submission format
  report.md
```

## Step 0 (now)
1. PLAN.md, decisions.md written, shown to user. Wait for go-ahead.

## Step 1: P1 seed-based (days 1-2)
1. Loader: read an instance (nodes, edges, seeds, truth) into sparse adjacency. Metrics module: precision, recall, F1, coverage, unseeded recall, macro-average over instances.
2. Baseline: degree / mean neighbor degree / clustering features, Hungarian on feature distance. Seed-free sanity check on val.
3. Seed method: seed-neighborhood signature. For node u in A, vector of adjacency to seed nodes (mapped to seed index in B). Similarity = overlap of seed neighbors (cosine/Jaccard, IDF-weighted) plus a small structural-feature term.
4. Iterative expansion: after Hungarian, accept high-confidence pairs as new pseudo-seeds (within this config only), recompute, repeat a few rounds. Final one-to-one Hungarian with a confidence rejection threshold.
5. Tune few hyperparameters (feature weight, rounds, accept margin, rejection threshold) on Train, check on Val. Freeze to config.json.
6. Run all 6 seed configs on Val (report), then on Test (predictions only). Output `predictions/<config>/<instance>/predictions.csv`, exactly `node_a,node_b`.
7. Analysis: by family, difficulty, size, seed strategy and fraction, confidence vs precision, runtime. README, report.md.
8. Validation report to user. Core item 1 done.

## Step 2: P3 membership inference (days 2-4)
1. Load digits.csv, dedupe pixels, stratified disjoint pools: target train/val/test 100 each, shadow pool 1200 (900 dev, 300 calibration). Save split ids to file.
2. MLP (64,32) via sklearn or a small numpy MLP. Fixed 600 epochs vs early stopping (val loss, patience 10, restore best). Needs per-epoch val loss and best-checkpoint restore, so I will use a numpy/sklearn `partial_fit` loop. Scaler fit on each model's own train members only.
3. 3 dev shadows (100 train, 100 val, 100 attack non-members each) plus 1 calibration shadow on the 300 reserved samples. Both modes (fixed, early stop) get their own shadows.
4. Attacks: loss threshold (score = -loss, threshold chosen on calibration shadow, frozen). Shadow logistic attack (features: true-class prob, loss, entropy, max prob, correct, task class one-hot). Model choices on calibration shadow only.
5. Evaluate on target: 100 members (target train) vs 100 nonmembers (target test). Target val excluded. Metrics: ROC-AUC, TPR@5%FPR (largest empirical TPR with FPR <= 0.05, no interpolation), TPR/FPR at frozen threshold, train/test acc.
6. Seeds 42, 43, 44, mean and sample std. ROC curves, learning curves, results tables, report.md.
7. Validation numbers to user. Core pair submittable.

## Step 3: P2 seed-free (days 5-6, only after steps 1-2)
1. Reuse P1 loader, metrics, features, Hungarian, rejection logic.
2. Degree-only baseline vs improved method (multi-hop neighborhood degree histograms, iterative refinement with confident pairs becoming pseudo-seeds, rejection threshold).
3. Tune on Train/Val, freeze, Test predictions. Report overall macro and by family and difficulty. No seed files read.

## Step 4: P4 DP reconstruction (days 6-7, hardest, only after step 3 or if time allows)
1. Read public.npz and validation npz. Verify g_t = (theta_t - theta_t+1)/0.1 against the known softmax gradient on non-private validation (sanity check).
2. Numpy softmax-gradient with analytic Jacobian for candidate images, Adam or L-BFGS, sigmoid-parameterized pixels in [0,1], summed over 10 steps.
3. Three attacks: plain, prior (class mean/var from public, prior strength tuned on validation), public class-mean baseline. B=1 non-private gets exact init from weight/bias gradient ratio.
4. Validation comparison across B and non-private / clip-only / eps 16, 4, 1. MSE, PSNR, vs public mean, convergence curves, figures.
5. Freeze, predict `observations/seed{42,43,44}/batch{1,2,4}_eps_4.npz`, one NPZ each, key `images` only, finite, in [0,1], order preserved.

## Cut lines if time runs short
- P1 then P3 are never cut.
- P2 is mostly a rerun of P1 code, so it should be cheap.
- P4 can ship with plain + mean baseline only if the prior attack is not finished; the report will say so.

## Notes
- projects.docx says choose 2 of the 4 projects and P3 and P4 are alternatives. You asked for all 4. Doing that, no scope change.
