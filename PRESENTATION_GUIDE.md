# Presentation Guide: 4 Privacy Projects

Read this top to bottom once. Each project has: the problem in plain English, then the 3 things you must present (how, why, evaluation).

## The big picture (say this first)
- All 4 projects are **privacy attacks**: given something an organization releases (an anonymized graph, a model's predictions, a model's updates), how much private information can an attacker recover?
- Projects 1 and 2: release an "anonymized" social graph. Can an attacker re-identify people?
- Project 3: release a trained model. Can an attacker tell whether a specific person's data was used to train it?
- Project 4: release model updates during training. Can an attacker rebuild the actual training images, and does differential privacy (DP) stop it?
- Rules of the assignment: tune only on Train/Validation, freeze settings, then run Test once. Test answers are hidden and graded by the instructor, so there are no Test scores to quote.
- Code lives in `privacy/p1_seed_based`, `p2_seed_free`, `p3_membership_inference`, `p4_dp_reconstruction`. Each has `report.md`, `config.json` (frozen settings), `src/`, `results/`.

---

## Project 1: Seed-based graph de-anonymization

### The problem
- Graph = people (nodes) and connections (edges). Graph A is an anonymized copy: names replaced with random IDs. Graph B is a separate, slightly different graph of the same people (some edges deleted, some added), also with random IDs.
- Goal: figure out which node in A is the same person as which node in B, using only the graph structure.
- **Seeds** = a few node pairs we already know are the same person (1%, 5%, or 10% of nodes). Chosen randomly or as the highest-degree nodes. 6 configs total.
- Data: 54 graph pairs (27 train, 9 val, 18 test), 3 graph types (Erdos-Renyi random, stochastic block = communities, preferential attachment = hubs), 300/600/1000 nodes.

### 1. How I did it
- Core idea (percolation): two nodes are probably the same person if they share many neighbors that are already matched. Score each pair (a in A, b in B) by counting matched common neighbors.
- Weighted so low-degree shared neighbors count more (1/log(2+degree)), then normalized by how many matched neighbors each node has.
- Add a tiny structural score (degree, avg neighbor degree, clustering, 2-hop walks), weight 0.02. It is only a tie-breaker.
- Loop: run Hungarian algorithm (best one-to-one assignment) on unmatched nodes, accept only confident pairs (at least 3 matched common neighbors, best in row and column) as new "pseudo seeds", repeat. If stuck, relax the threshold from 3 down to 1.
- Final Hungarian pass on whatever is left; drop pairs with score under 0.02. Seeds are always included in the output. Each seed config uses only its own seed file.
- Baseline for comparison: seed-free matching on degree, neighbor degree, clustering.
- Code: `src/match.py` (main), `src/features.py`, `src/baseline.py`, `src/metrics.py`.

### 2. Why this approach
- Seeds are strong anchors, so using "matched neighbors" directly exploits them. Simple and fast (0.05 to 1.3 seconds per graph).
- First version failed (F1 0.08 to 0.44): the structural score drowned out the seed signal and a margin rule blocked growth. Fix was strict common-neighbor counting plus a tiny structural weight.
- Chose this over graph embeddings because it is explainable and does not need training.

### 3. How I evaluated it
- Metrics: precision, recall, F1, coverage (fraction of nodes predicted), plus unseeded recall (only non-seed nodes, so seeds cannot inflate the score). Averaged per graph instance, not pooled.
- Tuned on Train only (27 graphs x 6 configs), froze `config.json`, ran Val once, then Test.
- Validation macro F1:

| seeds | random | high-degree |
|---|---|---|
| 1% | 0.909 | 0.924 |
| 5% (primary) | 0.991 | 0.987 |
| 10% | 0.998 | 0.996 |

- Seed-free baseline: 0.074 F1. Train mean 0.968 vs Val mean 0.967, so no overfitting.
- Takeaways: seeds matter enormously; 5% is already enough; 1% is fragile (a few early wrong matches snowball, worst instance 0.735); random vs high-degree is basically tied at 5%/10%.
- Honest limitation: Test submission is 108 prediction files (6 configs x 18 graphs), all pass the format checker, but I have no Test score.

---

## Project 2: Seed-free graph de-anonymization

### The problem
- Same task as Project 1 but with **no seeds**. Only the structure of A and B. Noise is smaller so it is feasible.
- Hard because many nodes look alike (same degree), so you cannot anchor anywhere.

### 1. How I did it
- Baseline (required): match by degree only (Hungarian on |deg_a - deg_b|).
- Step 1, spectral matching (GRAMPA): compute eigenvectors/eigenvalues of both adjacency matrices; nodes in the same position in the graph have similar eigenvector patterns. Build a similarity matrix from those, then Hungarian gives a full first matching.
- Step 2, refinement: 5 rounds of "count how many neighbors stay adjacent under the current matching", re-run Hungarian.
- Step 3, anchors plus percolation: pairs with at least 3 consistent neighbors (and at least 30% of the smaller degree) become pseudo seeds, then reuse the Project 1 percolation matcher to finish and reject weak pairs.
- Code: `src/seedfree.py`, reuses `p1_seed_based/src/match.py`. No seed file is ever read.

### 2. Why this approach
- Degree alone cannot separate nodes. Spectral methods use the whole graph shape, so they distinguish nodes with the same degree.
- Reusing Project 1's matcher gave a clean story: seed-free spectral step produces the "seeds", then the same engine finishes the job.

### 3. How I evaluated it
- Same metrics as Project 1, reported overall and by family and difficulty.
- Validation F1: **1.000 on all 9 graphs** vs 0.052 for degree-only baseline. Train was also 1.000 on all 27, so nothing needed tuning.
- A perfect score is suspicious, so I ran a leakage check: true mapping is not the identity, and randomly relabeling B's node IDs still gives 1.000. The method does not depend on ID values.
- Robustness probe (extra analysis, not the graded score): replace extra edges in B. Fine up to about 10% extra noise, collapses at 20% for random and community graphs (preferential attachment survives to 20%). So it works because this dataset's noise is mild, and it fails sharply, not gracefully.
- Limitation: Val cannot distinguish method variants since everything is 1.000.

---

## Project 3: Membership inference and early stopping

### The problem
- A model trained on a small dataset tends to **memorize** its training samples: it is more confident (lower loss) on them than on unseen data. An attacker who sees the model's output probabilities and the true label can guess "was this sample in the training set?" That is a privacy leak.
- Data: scikit-learn Digits (8x8 handwritten digit images, 10 classes). The label column is called `income` in the CSV but it is the digit 0-9.
- Two different labels: task label (the digit) and membership label (1 = was in training, 0 = not). Membership labels are built by me from the splits.
- Defense to test: **early stopping** (stop training when validation loss stops improving) so the model overfits less.

### 1. How I did it
- Split data into disjoint pools: target train 100, target val 100, target test 100, shadow pool 1200 (900 for developing the attack, 300 reserved for calibration). Per seed (42, 43, 44), fully independent repeat.
- Target model: MLP with hidden layers (64, 32). Two versions: **fixed 600 epochs** vs **early stopping** (val loss, patience 10, restore best checkpoint).
- Shadow models: the attacker's practice models. 3 dev shadows plus 1 calibration shadow, each with 100 train / 100 val / 100 non-member. They mimic the target so the attacker learns what members vs non-members look like.
- Attack 1, loss threshold: score = negative cross-entropy of the true label. Lower loss means more likely a member. Threshold chosen on the calibration shadow only.
- Attack 2, shadow logistic regression: features = true-class probability, loss, entropy, max probability, correct or not, task class. Trained on shadow members vs non-members. Regularization C picked by calibration AUC.
- Scaler fit only on each model's own training members (no leakage). Target validation excluded from the attack test. Attack test = 100 members vs 100 non-members.
- Code: `src/data.py` (splits), `src/mlp.py` (training loop), `src/attack.py` (attacks and metrics).

### 2. Why this approach
- Shadow models are the standard way to build a membership attack without touching the target's data (Shokri et al.). Loss threshold is the simplest strong baseline (Yeom et al.).
- Shadows use the same training mode as the target they attack, so the attacker adapts to early stopping.
- Manual `partial_fit` loop so early stopping and restore-best are exact. Batch size 32 set before any run, not tuned.
- Thresholds and C are chosen by code on the calibration shadow, never on target attack-test data.

### 3. How I evaluated it
- Metrics: attack ROC-AUC (0.5 = guessing), TPR at 5% FPR (how many members caught while falsely flagging at most 5 non-members), TPR/FPR at the frozen threshold, target train/test accuracy. Mean and sample std over seeds 42, 43, 44.

| mode | attack | test acc | AUC | TPR@5%FPR |
|---|---|---|---|---|
| fixed 600 | loss | 0.867 | 0.795 | 0.053 |
| fixed 600 | logistic | 0.867 | 0.833 | 0.210 |
| early stop | loss | 0.853 | 0.775 | 0.063 |
| early stop | logistic | 0.853 | 0.794 | 0.177 |

- The attack works (AUC 0.78 to 0.83, well above 0.5) because the model hits 100% train accuracy and train loss keeps shrinking while val loss does not.
- Logistic beats plain loss threshold, especially at low FPR.
- Early stopping: AUC drops by about 0.02 (loss) and 0.04 (logistic), test accuracy drops about 1.4 points. Small privacy gain, small utility cost. It is **not** a strong defense: TPR@5%FPR barely changes and early-stopped models still overfit (stopped at epoch 119/136/273).
- Limitations: only 100 vs 100 test samples, so TPR@5%FPR swings a lot between seeds (logistic fixed: 0.15, 0.32, 0.16). Wide error bars.

---

## Project 4: Reconstruction from model updates and differential privacy

### The problem
- Federated-style setting: a model is trained with a few SGD steps on a private batch of 1, 2, or 4 digit images, and the attacker sees the model weights after each of 10 steps. From the weight changes (the gradients) can the attacker rebuild the actual private images?
- Model: simple softmax linear classifier on 8x8 Digits (10x64 weights + 10 biases = 650 numbers). Labels are known to the attacker.
- Defenses: nothing; **clipping** (limit each example's gradient size to norm 1); **differential privacy** = clipping plus Gaussian noise, with total privacy budget epsilon in {16, 4, 1} covering all 10 steps. Smaller epsilon means more noise. Test files use epsilon = 4 only.
- Attacker recovers each step's gradient as $g_t = (\theta_t - \theta_{t+1}) / 0.1$.

### 1. How I did it
- Math fact used: gradient of one image = (softmax output minus one-hot label) times the image, a rank-one matrix. For batch size 1 without noise, one weight row divided by its bias gives the image exactly.
- Three attacks:
  - **Plain gradient matching**: optimize a candidate image (sigmoid keeps pixels in [0,1], Adam optimizer, 3 restarts) so its gradients match all 10 observed updates.
  - **Prior gradient matching**: same plus a penalty pulling the image toward the public class-mean image (mean and variance from the 800 public images). Weight = factor x sigma squared. Starts from the class mean.
  - **Public mean baseline**: ignore the updates, output the average image of the known class.
- Noise calibration from the assignment: $\rho = (\sqrt{L + \epsilon} - \sqrt{L})^2$ with $L = \log(1/\delta)$, $\sigma_{step} = \Delta \sqrt{10 / (2\rho)}$, $\Delta = 2C/B$.
- Verified first: my analytic gradient matches finite differences, and the noise in the validation files matches the stated sigma.
- Code: `src/grad.py` (gradients), `src/attack.py` (attacks), `tune.py`, `run.py`.

### 2. Why this approach
- Gradient matching is the standard gradient-inversion attack (Zhu et al., Geiping et al.). The linear model lets me write exact gradients by hand, so it is fast and checkable.
- The prior is the principled way to use public data: with factor 1 it is the Bayesian MAP estimate under Gaussian noise.
- The public-mean baseline matters because a digit-looking output could just be a class template, not real recovery.
- Tuned only on public validation images (lr, iterations, restarts, prior factor), then frozen before touching the Test observations. Never combined trials or seeds.

### 3. How I evaluated it
- Metrics: MSE and PSNR between reconstruction and original (lower MSE / higher PSNR is better), ratio vs the public-mean baseline, model accuracy after 10 updates, convergence plots, example images. Validation has public originals, so I can score there.
- No noise: private images are recovered. Batch 1 is exact (PSNR 86 dB). Batch 2 and 4: PSNR 43 and 39 dB with the prior attack. Clipping alone gives no protection.
- With DP: the prior attack collapses to the class template. Ratio to public mean is 0.986 to 1.004 everywhere (a coin flip). Plain matching is 6 to 7 times **worse** than the baseline (it fits noise).
- Why: clipped gradient has norm at most 1 spread across 650 numbers, but noise per number is about 2 to 8 at epsilon 4. Signal is far below noise.
- Privacy and utility tradeoff: the same noise destroys the model. At epsilon 4 with batch 1, accuracy falls from 0.96 to 0.21 after 10 steps.
- Honest result: at the Test setting (epsilon 4) the attack does **not** meaningfully beat the baseline. That is a legitimate finding (DP works), not proof reconstruction is impossible.
- Test output: 9 NPZ files (3 seeds x 3 batch sizes), format checker passes.

---

## Things to be ready for
- **Selection rule:** the assignment says pick 2 of 4, and Projects 3 and 4 are alternatives. You built all 4 because you were told extra projects earn extra credit. Be ready to say which pair is the "official" submission (core pair was P1 + P3) and confirm with the instructor how extras count.
- **AI use:** the code was AI generated. Check the course AI policy and add a disclosure if required. Be able to explain every method above in your own words.
- **No Test scores:** Test truth is hidden. Quote validation numbers only.
- **Likely questions and short answers:**
  - Why Hungarian algorithm? It gives the best one-to-one assignment, which the task requires.
  - Why do 1% seeds do worse? Too few anchors, early errors snowball.
  - Why is P2 perfect? Mild noise plus dense graphs; leakage check passed; it breaks at about 20% extra noise.
  - Why is AUC not near 1 in P3? Only 100 training samples and small test sets; the model generalizes decently (about 87% accuracy).
  - Why did DP stop the attack in P4? Noise scales with 2C/B per step and swamps the clipped gradient; cost is model accuracy.
