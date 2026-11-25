# P1 report: Seed-based graph matching

## Task and data
- Recover the one-to-one map between anonymized graph A and auxiliary graph B, given a seed file of known pairs.
- 27 Train, 9 Val, 18 Test instances. Families: Erdos-Renyi, stochastic block, preferential attachment. Difficulty easy / medium / hard = 300 / 600 / 1000 nodes and B edge retention about 0.935 / 0.835 / 0.735 (from `metadata.json`).
- Every instance has full node overlap (n_a = n_b, overlap 1.0), so a full matching is possible in principle.
- 6 seed configs per instance: random or high-degree selection at 1%, 5%, 10% (1% = 3, 6 or 10 seeds). 5% is the primary comparison.

## Method
- Score between unmatched a in A and b in B: matched-neighbor overlap. Count neighbors of a and b that are already matched to each other, IDF weighted by 1/log(2 + degree), normalized by sqrt((matched nbrs of a + 1)(matched nbrs of b + 1)).
- Plus a small structural kernel (weight `lam_struct` = 0.02) on standardized log degree, log mean neighbor degree, clustering and log 2-hop walk count. It is a tie-breaker and helps the first rounds at 1% seeds.
- Iterative expansion (percolation style), inside one seed config only:
  - Hungarian assignment on the unmatched block each round.
  - Accept an assigned pair as a pseudo seed when it has at least `min_common` matched common neighbors and is the best in its row and column (margin >= 0).
  - If nothing qualifies, relax `min_common` by 1, down to `min_common_floor` = 1. Stop when nothing qualifies at the floor or after 40 rounds.
- Final Hungarian on what is left, with rejection: final-assignment pairs with score < `reject_tau` are omitted. Seeds and pseudo seeds are always submitted.
- Seeds are fixed correspondences and are always in the output. No information passes between seed configs.
- Baseline for comparison: seed-free Hungarian on Euclidean distance of standardized (degree, mean neighbor degree, clustering).

## Frozen settings (`config.json`)
- min_common 3, min_common_floor 1, lam_struct 0.02, idf 1, rounds 40, bandwidth 1.0, accept_margin 0.0, reject_tau 0.02.
- All chosen on Train only (27 instances x 6 configs). Frozen before Val was run. Val was run once, then Test.
- Tuning history (Train macro F1, mean over 6 configs):
  - First version (margin 0.15, structural weight 0.5): F1 0.08 to 0.44. The structural term swamped the seed signal and the margin blocked expansion.
  - Strict common-neighbor acceptance without relaxation: 0.868. Floor 2: 0.952. Floor 1: 0.968.
  - lam_struct 0.1 hurt (about 0.88). Bandwidth 0.5 / 2.0: 0.964 / 0.955 vs 0.968 at 1.0. Rounds 20, 40, 80 identical (expansion converges well before 20).
  - reject_tau 0.02 vs 0: precision 0.9691 vs 0.9684, recall unchanged 0.9684, coverage 0.9992. Higher tau (up to 0.3) gave the same numbers. The gain is tiny because almost every node becomes a pseudo seed and only a handful reach the final assignment.

## Val results (9 instances, macro average per instance)
Source: `results/val_macro_by_config.csv`, `results/val_per_instance.csv`.

| config | precision | recall | F1 | coverage | unseeded recall | F1 sample std | runtime s |
|---|---|---|---|---|---|---|---|
| random 1% | 0.911 | 0.907 | 0.909 | 0.996 | 0.906 | 0.082 | 1.05 |
| random 5% | 0.991 | 0.991 | 0.991 | 1.000 | 0.990 | 0.012 | 0.42 |
| random 10% | 0.998 | 0.998 | 0.998 | 1.000 | 0.998 | 0.003 | 0.36 |
| degree 1% | 0.925 | 0.923 | 0.924 | 0.997 | 0.922 | 0.090 | 0.61 |
| degree 5% | 0.987 | 0.987 | 0.987 | 1.000 | 0.986 | 0.019 | 0.43 |
| degree 10% | 0.996 | 0.996 | 0.996 | 1.000 | 0.996 | 0.004 | 0.27 |
| seed-free baseline | 0.074 | 0.074 | 0.074 | 1.000 | 0.074 | | 0.13 |

- Val numbers are close to Train (Train mean 0.968, Val mean over the 6 configs 0.967), so there is no sign of overfitting the few tuned settings.
- Primary comparison (5%): random 0.991, degree 0.987. The difference is within one std across instances.
- Unseeded recall is almost the same as recall, so the score is not driven by the seeds themselves.

### By family (macro F1)
| family | rnd 1% | rnd 5% | rnd 10% | deg 1% | deg 5% | deg 10% | baseline |
|---|---|---|---|---|---|---|---|
| Erdos-Renyi | 0.907 | 0.987 | 0.998 | 0.926 | 0.984 | 0.999 | 0.045 |
| pref. attachment | 0.915 | 0.992 | 0.999 | 0.983 | 0.995 | 0.996 | 0.111 |
| stochastic block | 0.904 | 0.993 | 0.997 | 0.862 | 0.981 | 0.994 | 0.065 |

### By difficulty (macro F1)
| difficulty | rnd 1% | rnd 5% | rnd 10% | deg 1% | deg 5% | deg 10% | baseline |
|---|---|---|---|---|---|---|---|
| easy | 0.987 | 0.996 | 1.000 | 0.908 | 1.000 | 0.997 | 0.173 |
| medium | 0.840 | 0.998 | 0.999 | 0.988 | 0.994 | 0.997 | 0.036 |
| hard | 0.900 | 0.979 | 0.995 | 0.876 | 0.966 | 0.995 | 0.012 |

- Each cell is 3 instances (one per family), so these breakdowns are noisy. Per-instance values are in `results/val_per_instance.csv`.

### Confidence and runtime
- Pair kinds (`results/val_precision_by_pair_kind.csv`): pseudo seeds are 0.897 / 0.987 / 0.997 precise for random 1 / 5 / 10%. Only 6 to 9 pairs per 1% config reach the final assignment, and all of them are wrong; at 5% and 10% none do.
- Confidence vs precision (`results/val_confidence_vs_precision.csv`, figure `val_confidence_vs_precision.png`, pooled per config): errors sit in the lowest score quintile. Random 1%: quintile 1 precision 0.497, quintiles 2 to 5 at 0.991 or higher. Random 5%: 0.934 then 1.000. The score is a useful confidence signal.
- Runtime (`val_runtime.png`): mean 0.05 s / 0.27 s / 1.26 s for 300 / 600 / 1000 nodes (dense float32 matrices, one Hungarian per round). Baseline 0.01 / 0.06 / 0.33 s.

## Analysis
- **Seeds matter a lot.** The seed-free structural baseline reaches only 0.074 F1 (0.012 on hard). Degree, mean neighbor degree and clustering are far too coarse to tell nodes apart, especially in Erdos-Renyi graphs where degrees concentrate.
- **5% seeds are enough** for near-complete recovery on every family and difficulty (lowest instance F1 0.950, stochastic block hard, degree seeds). 10% adds a little on hard instances.
- **1% seeds are the fragile regime.** With 3 to 10 seeds, the first pseudo seeds must come from relaxed acceptance (min_common relaxing to 1) plus the structural tie-breaker. A few early wrong pairs then seed more wrong pairs. This is where the large instance-to-instance spread comes from (F1 std 0.08 to 0.09, worst instance 0.735).
- **Random vs high-degree.** At 5% and 10% the two are tied. At 1% neither wins consistently: degree seeds are better on preferential attachment (0.983 vs 0.915; hubs touch many nodes, so they give many matched neighbors early) and worse on stochastic block (0.862 vs 0.904; high-degree seeds tend to cluster in the same blocks and leave other blocks unanchored). On a 9-instance Val set these differences are within noise.
- **Hard structures.** Hard instances (1000 nodes, about 26% of B edges replaced) are the lowest at 5% (0.979 / 0.966). Stochastic block graphs are the hardest at 1% degree seeds. Preferential attachment is easiest overall because degree heterogeneity makes both the overlap score and the structural term more discriminative.

## Test submission
- `predictions/<seed_config>/<instance>/predictions.csv`, 108 files. `run.py check` reports 108 / 108 valid: header exactly `node_a,node_b`, integer ids that exist in the node files, no duplicate node_a or node_b, every seed pair of that config included.
- Coverage: 299 to 1000 pairs per file, a few low-confidence final pairs omitted by the rejection threshold (`results/test_prediction_log.csv`).
- Test truth is hidden. No Test numbers are reported here.

## Limitations
- Val has 9 instances (one per family and difficulty), so breakdown cells are single instances and the random vs degree comparison has low power.
- Rejection is weak. The frozen reject_tau 0.02 only touches the final-assignment pairs, while most errors at 1% are low-score pseudo seeds that are always submitted. The Val confidence table shows that dropping the lowest-score quintile at 1% would raise precision a lot at the cost of recall. That was not tuned or applied, because the config was frozen before Val and changing it after seeing Val would mix tuning and reporting.
- Dense n x n matrices: fine for 1000 nodes, would need sparse candidate lists for much larger graphs.
- All instances have full node overlap. Behaviour with partial overlap (nodes only in one graph) is untested; the rejection threshold would matter more there.
- The structural features are simple; embeddings or multi-hop fingerprints were not tried.
- Found later, in P2: a seed-free spectral method (GRAMPA plus refinement) reaches F1 1.000 on all Train and Val graphs of the same generators. Using it to bootstrap P1 at 1% seeds would likely remove the 1% weakness. P1 was not changed, because its config was frozen and Test predictions were already written.
