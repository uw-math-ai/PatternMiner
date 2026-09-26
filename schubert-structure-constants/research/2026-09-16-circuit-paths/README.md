# Circuit analysis of the Schubert witness selector

Completed September 17, 2026. The experiments identify a causal contribution from inverse positions at indices outside the candidate set, distinguish the roles of several receiving attention heads, and show that the resulting signal depends on distributed later computation. They do not identify an exact rule for Schubert multiplication.

The user explicitly requested proceeding with circuit analysis on the existing checkpoint. [PROTOCOL_AMENDMENT.json](PROTOCOL_AMENDMENT.json) removes the earlier accuracy prerequisite. The checkpoint remains `19d12b2bcdb0ad6ccf4f40dd3e2ed89662beab425449daf7808af05785500894`; its earlier S10 assessment remains 1,874/1,902 correct difficult pairs. No training, coefficient queries, or manuscript changes were made.

## Task and samples

For ordinary Schubert structure constants, write

\[
A_i=c^w_{s_i u,v},\qquad B_j=c^w_{u,s_jv},\qquad
\ell(w)=\ell(u)+\ell(v)-1.
\]

Given a supplied index with \(A_i>0\), the task is to choose \(j\in D_L(v)\) with \(B_j>0\). Every pair in these experiments has the same factors and candidate set, but different outputs \(w,w'\) and disjoint sets of valid answers. Both directions of replacement are measured. Layer and head numbers are zero-based.

| Sample | Pairs | Factor orbits | Pairs with both native answers correct |
|---|---:|---:|---:|
| Preassigned discovery | 320 | 29 | 316 |
| Preassigned confirmation | 249 | 22 | 248 |
| Earlier development pairs with complete candidate data equal | 8 | 4 | 8 |

Discovery contains 95 pairs from the S9 factor population and 225 from S10; confirmation contains 178 and 71 respectively. These are finite sampled populations, not exhaustive symmetric groups. Stable outputs can require a larger symmetric group. All available mined pairs fit within the predetermined cap of 512. Discovery and confirmation factor orbits are disjoint. Neither sample was used to train the checkpoint or select the earlier neuron ranking.

The first two samples match the original 22 candidate features, but none matches all augmented candidate features. The final eight pairs match all 148 candidate features and all 78 candidate-to-candidate relation channels. Results from these cohorts are reported separately. The eight-pair cohort is earlier development evidence.

## Confirmed components and input dependence

Scanning all 96 heads and six MLP outputs on discovery again nominates **L1H0/H8 and L3H3/H11**. On confirmation, their joint output replacement changes the score contrast by **15.738**, compared with **4.115** for the mean of twenty controls matching layers and the pairing inherited from widening. Their paired difference is **11.623**, with a descriptive factor-orbit bootstrap interval **[8.826, 14.920]**. Ablating these heads leaves **215/249** pairs correct. Retaining only these heads leaves **80/249** correct, with all MLPs and residual paths still active. They contribute substantially but do not constitute a sufficient circuit.

The previously fixed first-MLP ranking also survives the preassigned confirmation test. The primary comparison, top 16 unit pairs minus the mean of twenty fixed magnitude-matched random subsets, is **2.817**, with 95% factor-orbit bootstrap interval **[1.410, 4.533]**. The top 32 pairs give effect **8.471**, versus **11.658** for all 384 pairs, along the path from noncandidate first-MLP activations through L1 keys and values. The ratio, 72.7%, concerns these mean intervention effects; it is not a fraction of accuracy or of a multiplication algorithm.

The preactivation decomposition identifies which input changes drive this path:

| Input contribution, in the fixed 32 unit pairs | Mean contrast effect |
|---|---:|
| Inverse positions \(w^{-1}(j),w^{-1}(j+1)\) | 10.124 |
| Endpoint with the larger absolute effective coefficient in each neuron | 9.587 |
| Signed difference \(w^{-1}(j+1)-w^{-1}(j)\) | 6.219 |
| Left-descent indicator | 0.084 |
| Rank-matrix columns | 0.211 |
| Relative-root features | 0.470 |

Positions are divided by the ambient size in the encoding. The signed difference is the height of \(w^{-1}\alpha_j\). The other input contributions and normalization are held as specified in the decomposition. Effects after the activation function and subsequent layers are not additive; in particular, an individual input contribution may exceed the net path effect. This identifies sensitivity to inverse positions, rather than an exact formula for what each neuron computes.

## Complete enumeration of receiving-head subsets

We evaluated **all 256 subsets of the eight L1 head pairs**, separately for keys, values, and their joint replacement, on all 1,138 directed discovery and confirmation inputs. Each subset receives the same fixed first-MLP perturbation. This gives exact Shapley attributions for these eight groups, without sampling subsets.

In both cohorts, **H0/H8 has the largest key attribution**, and **H6/H14 has the largest value attribution**. On confirmation these are **3.356 of 5.770** for keys and **1.132 of 3.974** for values. These roles overlap: H0/H8 also contributes through its values. Other head pairs make substantial contributions.

The joint key/value effect exceeds the sum of the eight isolated-pair effects by **2.248**, with descriptive interval **[1.073, 3.590]**. Thus isolated ablations miss interactions. A subset of seven pairs can approximate the effect reasonably, but the best seven-pair discovery subset has 8.3% relative contrast discrepancy and 97.66% agreement with choices under the full intervention. The predetermined 10% discrepancy and 99% choice-agreement rule retains all eight pairs. Its subsequent perfect replay is the identity control, not a compression result.

This enumeration was designed after the first confirmation results were inspected. It is an exploratory follow-up; the second cohort is not a new blind confirmation for these adaptive questions.

## Later computation and an explicit matched example

With the top 32 unit pairs transmitted through all L1 heads, the confirmation mean effect is **8.471**. Restoring later MLP outputs to their native destination values reduces it to **1.511**; restoring later attention outputs reduces it to **2.101**; restoring both reduces it to **0.435**. These interventions hold a component's response fixed, rather than zeroing it. They measure different counterfactual computations and are not an additive partition.

The isolated serial path through L1 candidate MLP outputs, L3H3/H11, and L3 candidate MLP outputs gives only **0.048**, with interval **[-0.023, 0.114]**. The results therefore do not support that short chain as the explanation of the larger path effect.

A separate experiment transmits L1 outputs only at noncandidate positions, holds intervening attention outputs fixed, and allows a later layer to read those positions through its keys and values. Candidate residual states are checked to remain exactly unchanged before that receiving layer. The strongest direct second transfer is through **L2**, with confirmation effect **1.034 [0.491, 1.687]** for the fixed top 32 unit pairs. It explains a contribution, not the whole decision.

On the eight development pairs with identical complete candidate data, the full first-MLP-to-L1-key/value intervention changes the answer to the donor's valid index in **12/16 directions**; the fixed top 32 pairs do so in **7/16**. All eight native pairs are correct. Every case, including directions without a switch, is retained in [MATCHED_CASES.json](MATCHED_CASES.json).

For the previously recorded example,

\[
\begin{aligned}
u&=(1,3,8,2,6,4,7,5,9,10),\\
v&=(1,6,2,3,7,5,8,4,9,10),\\
w&=(1,10,5,2,6,4,8,3,7,9),\\
w'&=(2,10,5,1,6,4,7,3,8,9),
\end{aligned}
\]

the candidates are 4 and 5, and the supplied index 5 is valid for both inputs. The valid answer is 5 for \(w\), and 4 for \(w'\). The model answers both correctly. Transferring all first-MLP activations outside the candidate set from \(w\) to \(w'\), solely through L1 keys and values, changes the latter answer from 4 to 5. The reverse transfer does not change the former answer. This gives a concrete causal use of information absent from the complete candidate representation.

## Interpretation and next experiment

The evidence supports a distributed computation: learned functions sensitive to inverse positions at other indices alter attention keys and values; subsequent attention and positionwise nonlinear transformations use the resulting signal to distinguish candidate indices. It does not yet identify a Bruhat-order test or an exact criterion for \(B_j>0\).

The next useful target is an explicit description of the inverse-position comparisons used by L1H0/H8 and the information carried by L1H6/H14 and L1H2/H10. Test any proposed description by replacing the corresponding computation, measuring both score discrepancies and witness choices, and inspecting the failures. The short-chain hypothesis tested here should not be used as a premise for an algebraic conjecture.

## Artifacts and validation

- [RESULTS.json](RESULTS.json): numerical summary, sample counts, and comparisons.
- [CIRCUIT_RESULTS.pdf](artifacts/CIRCUIT_RESULTS.pdf): figure; intervals are 95% percentile bootstraps over factor orbits. The 22 confirmation orbits limit precision. Adaptive comparisons are not adjusted for multiplicity.
- [HEAD_CONFIRMATION.json](HEAD_CONFIRMATION.json), [MLP_CONFIRMATION.json](MLP_CONFIRMATION.json), and [ENDPOINT_CONFIRMATION.json](ENDPOINT_CONFIRMATION.json): fixed component and input tests.
- [COALITIONS_CONFIRMATION.json](COALITIONS_CONFIRMATION.json): every receiving-head coalition, exact attributions, and interactions.
- [PATH_TRACE_ALL_HEADS.json](PATH_TRACE_ALL_HEADS.json), [MATCHED_PATH_TRACE.json](MATCHED_PATH_TRACE.json), and [SECOND_HOP.json](SECOND_HOP.json): successive path and blocking experiments.
- [REPRODUCE.md](REPRODUCE.md): execution order and data dependencies.

Native versus explicit forward computation, identity replacement, complete input restoration, and parameter-gradient checks pass exactly. Preactivation reconstruction has maximum error below \(1.3\times10^{-5}\); replaying its decomposed intervention differs by at most \(8.1\times10^{-5}\). Exact coalition attributions satisfy their efficiency identity to numerical precision. All six earlier evidence manifests and the manuscript PDF were verified unchanged.
