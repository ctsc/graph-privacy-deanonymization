# P2 report: Seed-free graph matching

## Task and data
- Same graph families, sizes and noise levels as P1 (27 Train, 9 Val, 18 Test), but no seeds. Structure only.
- All instances have full node overlap (n_a = n_b). B edge retention about 0.935 / 0.835 / 0.735 for easy / medium / hard.

## Method
- **Baseline (required):** degree only. Hungarian on |deg_a - deg_b|.
- **Step 1, spectral initial matching (GRAMPA, Fan, Mao, Wu, Xu 2019):**
  - Eigendecompose A / sqrt(mean degree) and B / sqrt(mean degree): A = sum l_i u_i u_i^T, B = sum m_j v_j v_j^T.
  - Similarity X = sum_ij w_ij (u_i^T 1)(v_j^T 1) u_i v_j^T with w_ij = 1 / ((l_i - m_j)^2 + eta^2), eta = 0.2.
  - Hungarian on X gives a full one-to-one matching. Fully label-free and permutation equivariant.
- **Step 2, refinement:** 5 rounds of S = A P B^T (number of neighbors kept adjacent by the current matching P), Hungarian on S.
- **Step 3, anchors plus percolation:** pairs with at least 3 consistent neighbors and at least 30% of the smaller degree become anchors. They are fed as pseudo seeds into the P1 percolation matcher (same settings as P1's frozen config), which re-matches the rest and applies the rejection threshold.
- Runtime is dominated by two dense eigendecompositions: about 0.1 s / 0.7 s / 2 to 3 s for 300 / 600 / 1000 nodes.

## Settings and freezing
- `config.json` values were set before the first run (eta 0.2 is a typical GRAMPA choice; percolation settings copied from P1's frozen config). They were checked on Train only: macro F1 1.000 on all 27 Train instances, for the spectral step alone and for the full pipeline. Nothing was tuned, because there was nothing left to improve on Train.
- Frozen before Val. Val run once, then Test.

## Leakage check
- A perfect score is suspicious, so before Val I checked on 3 hard Train instances:
  - Ground truth is not the identity (0.0 to 0.2% of nodes have the same id in A and B).
  - Randomly relabeling B (and the truth with it) still gives F1 1.000. The method does not depend on node ids.

## Val results (9 instances, macro average per instance)
Source: `results/val_macro.csv`, `results/val_per_instance.csv`.

| method | precision | recall | F1 | coverage | runtime s |
|---|---|---|---|---|---|
| degree baseline | 0.052 | 0.052 | 0.052 | 1.000 | 0.008 |
| spectral initial only | 1.000 | 1.000 | 1.000 | 1.000 | 0.88 |
| spectral + refine + percolation (submitted) | 1.000 | 1.000 | 1.000 | 1.000 | 0.78 |

- Every Val instance is matched perfectly by the submitted method (all 9 family x difficulty cells at 1.000).
- Degree baseline by difficulty: easy 0.078, medium 0.051, hard 0.028. By family: preferential attachment highest (0.120 easy), stochastic block lowest (0.017 hard).
- Confidence: all 5,700 Val pairs ended as anchors (consistency test passed), all correct. With a perfect result there is no confidence vs precision curve to show; the robustness probe below gives the informative picture.

## Robustness probe (analysis only, config unchanged)
- `robustness.py`: on each Val instance, replace an extra fraction f of B's edges with random non-edges (same delete plus replace protocol), then rerun the frozen method. Source `results/val_robustness.csv`, figure `results/val_robustness.png`.

| extra replaced f | easy | medium | hard | ER | PA | SBM |
|---|---|---|---|---|---|---|
| 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.1 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 |
| 0.2 | 0.348 | 0.330 | 0.336 | 0.014 | 0.991 | 0.009 |
| 0.3 | 0.041 | 0.285 | 0.007 | 0.004 | 0.323 | 0.007 |
| 0.4 | 0.018 | 0.004 | 0.001 | 0.003 | 0.018 | 0.002 |

- The method has a sharp breakdown point, typical of spectral matching: fine up to about 10% extra replacement, then collapses for Erdos-Renyi and stochastic block graphs at 20%.
- Preferential attachment survives longer (0.991 at 20%): hubs give strong, distinctive spectral and neighborhood signals.
- Refinement plus percolation matters near the edge: at f = 0.1 on easy graphs the spectral step alone gives 0.807 and the full pipeline 0.998.
- When the spectral step fails, the anchor test lets few pairs through and percolation cannot recover; the method does not fail gracefully.

## Analysis
- **Why it works so well here:** the graphs are dense (Erdos-Renyi p = 0.06, BA m = 6, SBM p_in 0.13) and the supplied noise is at most about 26.5% replaced edges. Under that regime the leading spectral structure of A and B is close enough for GRAMPA to align almost every node, and one round of neighbor-consistency refinement fixes the rest.
- **Degree alone is useless:** 5% F1. Many nodes share a degree, and edge noise shuffles the order.
- **Family and difficulty:** no difference on the supplied data (all 1.000). Differences only appear under extra noise (above).

## Limitations
- 9 Val instances, all at 1.000: Val cannot separate method variants, and Test performance on the same generators is expected to be similar but is not guaranteed.
- Dense n x n eigendecompositions: O(n^3), fine for 1000 nodes, too slow for very large graphs.
- Sharp failure beyond the noise level of this dataset (see probe). Partial node overlap was not tested (all instances have full overlap).
- The robustness probe uses my own perturbation, not the teacher generator, so it is only indicative.
