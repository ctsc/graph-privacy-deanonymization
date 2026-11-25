# decisions.md

Running log. Newest at bottom of each section. Method choice per project is locked here before coding.

## Global
- D1 (setup, superseded by D4): Use system Python 3.14 with numpy/scipy/pandas/sklearn/matplotlib. The `privacy_projects` conda env is not installed and torch/networkx are absent. Avoid both. Versions differ slightly from `environment.yml` pins (numpy 2.4.0 vs 2.4.6, pandas 2.3.3 vs 3.0.6, sklearn 1.8.0 vs 1.9.1). Will state in READMEs.
- D2 (scope): `projects.docx` says pick 2 of 4 (P3 and P4 are alternatives). Per your instruction, all 4 get built, core pair first.
- D3 (git): No repo yet. When you create one, commits authored by you only, no Co-Authored-By, no AI attribution. This overrides the default commit trailer.

- D4 (setup): Installed conda env `privacy_projects` from `environment.yml` (numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, sklearn 1.9.1, networkx 3.6.1) plus matplotlib 3.11.2 via pip. All code runs in this env. matplotlib is not in `environment.yml`, so READMEs will list it as an extra.

## Method choice per project (proposed, pending your OK)
- P1: seed-neighborhood signature similarity + light structural features, Hungarian, iterative pseudo-seed expansion within a single config, confidence-based rejection. Not embeddings (simple, fast, explainable).
- P2: same code minus seeds. Degree-only Hungarian baseline vs multi-hop degree-histogram features with iterative refinement and rejection.
- P3: numpy/sklearn MLP (64,32) with manual epoch loop, loss-threshold attack and logistic shadow attack, both fixed-600 and early-stopping variants.
- P4: numpy analytic-gradient matching with Adam, sigmoid pixel parameterization, class-mean/var Gaussian prior, public-mean baseline.

## Open questions (will ask before proceeding on each)
- Q1: P1 `predictions.csv` layout for 6 configs x 18 test instances: proposed `predictions/<seed_config>/<instance>/predictions.csv`. OK?
- Q2: P3 MLP implementation: sklearn `MLPClassifier` with `partial_fit` (Adam, default lr) vs hand-written numpy. Proposed sklearn partial_fit. OK?

## Log
- (empty, filled during build)
- D5 (Q1 resolved, default): P1 predictions at `p1_seed_based/predictions/<seed_config>/<instance>/predictions.csv`, seed pairs included in predictions.
- D6 (Q2 resolved, default): P3 uses sklearn MLPClassifier with a manual partial_fit epoch loop. Not started.
- D7 (P1 method, as built): percolation-style seed matching. Score = matched-common-neighbor overlap (IDF weighted, normalized by sqrt of matched-neighbor counts) + small structural kernel (lam_struct). Hungarian each round. Accept a pair as pseudo-seed if margin >= 0 and matched common neighbors >= min_common. When nothing qualifies, min_common relaxes by 1 down to min_common_floor. Final Hungarian with rejection tau. All inside one seed config. Dense float32 matrices, ~0.1 to 0.6 s per instance.
- D8 (P1 finding): the first version (margin 0.15, struct weight 0.5) gave F1 0.08 to 0.44 because the structural term swamped the seed signal and the margin blocked all expansion. Strict acceptance on common-neighbor counts (percolation) was the fix. Struct weight 0.02 is a tie-breaker and bootstrap aid for 1% seeds, 0.1 hurts.
- D9 (P1 metrics): precision, recall, F1, coverage computed on the full submission including seed pairs. unseeded_recall and unseeded_precision exclude seed nodes. Coverage = predicted pairs / n_a. All macro averaged over instances.
- D10 (tuning scripts): saved in p1_seed_based/tuning/ (grid scripts) and p1_seed_based/tune_util.py (parallel evaluator, Train/Val only). Note: grid_floor_lam.py is the latest. Scripts use sys.path hacks and expect cwd = p1_seed_based.
- D11 (resolved): reject_tau = 0.02. Train sweep 0 to 0.3: precision 0.9684 to 0.9691, recall unchanged, coverage 0.9992; every tau >= 0.02 identical. Rounds 20/40/80 identical (kept 40). Bandwidth 0.5/2.0 worse than 1.0. Script tuning/sweep_tau_rounds_bw.py.
- D12 (P1): config.json frozen after D11, before Val. Val run once, then Test. The Val confidence table later showed that dropping low-score pseudo seeds at 1% would raise precision; not applied (would mean tuning on Val after freezing). Stated in report.
- D13 (P1): baseline = seed-free Hungarian on standardized (degree, mean neighbor degree, clustering). Test predictions include seed pairs; checker in src/check_predictions.py.
- D14 (P3): the Digits CSV label column is named `income`; it holds the digit classes 0-9. Pixel dedupe removed 0 rows. Seed controls pools, MLP init and batch order (full independent repeat per seed). 297 records unused.
- D15 (P3): MLP batch_size 32 (sklearn default would be one full-batch Adam step per epoch on 100 samples), alpha 1e-4, lr 1e-3. Set before any run, not tuned. Shadows use the same training mode as the target they attack. Attack nonmembers for shadows are the shadow's own 100 attack-nonmember records (shadow val excluded, mirroring the target).
- D16 (P3): logistic C chosen from {0.01, 0.1, 1, 10} by calibration-shadow AUC; thresholds for both attacks = largest calibration TPR with FPR <= 0.05. Chosen per seed by code, frozen before target evaluation.
- D17 (P2): method = GRAMPA spectral similarity (eta 0.2) + 5 rounds A P B^T refinement + consistency anchors + P1 percolation. Defaults set before the first run; Train F1 1.000, so nothing tuned. Leakage check done (truth not identity; random relabeling of B still 1.000). Robustness probe on Val is analysis only.
- D18 (P4): noise model verified on public validation (residual std = sigma_per_step). Plain attack uses random inits (plus exact ratio init for B1 sigma 0); prior attack inits at the class mean. Prior weight factor * sigma^2 per spec, factor 1.0 chosen on DP validation (equals MAP). No prior floor for sigma = 0 (validation showed it hurts). Submitted attack chosen by eps 4 validation MSE: prior 0.044743 vs mean 0.044744, a tie; prior kept by the pre-set rule.
- D19 (workflow): user asked to finish the whole PLAN without pausing between phases, so P2 and P4 followed P1 and P3 directly (overrides the earlier "wait for go-ahead after P1").
