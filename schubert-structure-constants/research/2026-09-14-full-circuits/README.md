# Full training and circuit analysis

This continuation starts from commit `[withheld for anonymous review]`. It uses the previously approved extension of four local T4 GPU-hours, six total timed hours and 20,000 exact-product queries. The resource agreement and cumulative ledger are retained here. The manuscript and all five earlier evidence manifests have been verified unchanged.

The full training and development circuit analysis is complete. The final witness selector passes all twenty-seven development criteria, with minimum accuracy **99.303%**, but fails the replacement independent assessment on difficult S10 pairs: **1,874/1,902 = 98.528%**, despite **99.799%** accuracy on individual supplied inputs. S7, S8 and S9 pass. The required independent 99% threshold remains unmet; the original formal discovery and confirmation populations remain protected. All circuit results in this directory are development evidence.

## Mathematical task and data

For permutations satisfying the recorded Bruhat hypotheses and \(\ell(w)=\ell(u)+\ell(v)-1\), put

\[
A_i=c^w_{s_i u,v},\qquad B_j=c^w_{u,s_j v}.
\]

Given an index with \(A_i>0\), the task is to select a left descent \(j\) of \(v\) with \(B_j>0\). The retained network structurally discards the supplied index. Qualification requires at least 99% separately on every declared population, including both orientations, every positive supplied index, and simultaneous correctness on both members of difficult pairs.

Population labels S7–S10 specify the sampled factors. Stable products can have outputs in a larger symmetric group; retained output permutations have size at most seventeen. The length bounds and sampling distributions are recorded in `DATA_PLAN.json` and `NEIGHBOR_PLAN.json`.

New assessment factors were reserved before development and training factors. All selections exclude the retained factor orbits and the previously eligible symmetry transforms. The fresh development populations contain 31,940 oriented S9 examples and 29,400 oriented S10 examples, with 177 and 574 difficult pairs. Together with earlier development populations, these give nineteen selection criteria. Previously failed assessments remain development data and are excluded from training.

Exact generation near the remaining development errors produced **4,683,591 new examples and 27,352 difficult pairs in 952 factor groups**, using 9,000 exact-product queries. The resulting training corpus has **16,738,276 examples**. [The tensor audit](NEIGHBOR_TRAINING_AUDIT.json) checks the actual concatenated fields, source offsets and 1,024 sampled new pairs. [The cache audit](CACHE_ARCHIVE_VERIFICATION.json) verifies every compressed shard and the reconstructed byte streams.

The [first training continuation](neighbor_refinement.json) completed 12,000 updates and selected update 10,500. The [registered readout comparison](FINAL_REFINEMENT_READOUTS.json) selected the original score plus twice the model's prediction of log(1+B_j), merged into the ordinary linear output head. These are learned predictions; exact coefficients are not model inputs. [Native evaluation](FINAL_NATIVE_VALIDATION.json) verifies the saved network on all nineteen criteria. [Its selection](FINAL_MODEL_SELECTION.json) freezes its hash before [independent assessment](FRESH_ASSESSMENT.json). The failed assessment retains its predictions and its explicit exposure record.

The [second continuation plan](SECOND_CONTINUATION_PLAN.json) appends eight exposed development criteria, giving twenty-seven in total. It combines further exact supervision with a penalty for departing from an earlier model's candidate distribution, applied only where that earlier model is exactly correct on the training input. It then adds disjoint neighbors of eleven newly exposed failure factor orbits. [Replacement assessment factors](SECOND_DATA_PLAN.json) were reserved before selecting these neighbors. Neither assessment's rows are added to training. These are adaptive phases; they do not separately estimate the causal effects of the teacher penalty and new data.

## Final selected model and its analysis

The second continuation completed 6,000 teacher-regularized updates and 9,000 updates on the enlarged 17,994,585-row corpus. The initial memory allocation failure and its charge are retained. Range-checked signed 16-bit labels saved 1,727,480,160 GPU bytes without changing their values, batch size, sampling or loss.

The [registered parameter comparison](SECOND_INTERPOLATION.json) selected 25% of the retained update-1,500 parameters and 75% of the training-selected update-7,500 parameters. The [readout comparison](SECOND_REFINEMENT_READOUTS.json) then added eight times the learned prediction of log(1+B_j) to the selector score, merging it into a single linear output head. The model does not receive exact coefficient values. [Native verification](SECOND_NATIVE_VALIDATION.json) reproduces all twenty-seven development criteria before [selection](SECOND_MODEL_SELECTION.json) freezes the checkpoint.

The final checkpoint is [SECOND_REFINEMENT_READOUTS.pt](artifacts/SECOND_REFINEMENT_READOUTS.pt), SHA-256 `19d12b2bcdb0ad6ccf4f40dd3e2ed89662beab425449daf7808af05785500894`. The [replacement assessment](SECOND_ASSESSMENT.json) gives:

| Sampled factors | Supplied-input accuracy | Difficult pairs correct |
|---|---:|---:|
| S7 | 100% | 1/1 |
| S8 | 99.943% | 26/26 |
| S9 | 99.935% | 49/49 |
| S10 | 99.799% | 1,874/1,902 (98.528%) |

Each pair requires simultaneous correctness for both members at every positive supplied index. These are finite samples with the bounds in SECOND_DATA_PLAN.json. [The exposure record](SECOND_ASSESSMENT_EXPOSURE.json) prevents reuse as independent evidence after adaptation. The first independent failure, 1,278/1,308 S10 pairs, remains in FRESH_ASSESSMENT.json; the two assessments have different factor populations and are not a matched estimate of improvement.

[The full component scan](FINAL_COMPONENT_SCAN.json) measures all 96 attention heads and six MLPs on all **5,445 development pairs in 500 factor orbits**, retaining gradient approximations and direct activation replacements. The unmodified final model solves 5,436 of these pairs. Every component is reported; rankings are exploratory.

[The fixed MLP-path replay](FINAL_DEVELOPMENT_DECOMPOSITION.json) uses the earlier neuron ranking and twenty magnitude-matched random controls. On 447 factor orbits excluded from that ranking, the fixed 32 hidden-unit pairs give mean effect 10.1313, compared with 15.0417 for the full path: **67.35% of the mean intervention effect**. The fixed 16-pair effect exceeds the mean random-control effect by 4.3764, with exploratory group-bootstrap interval [3.9551, 4.8234]. This is not a percentage of an algorithm or of accuracy explained.

[Final same-support controls](FINAL_SUPPORT_CONTROLS.json) use the unchanged 1,247-input, 202-orbit donor cohort. For 32 hidden-unit pairs, opposite-support and same-support donors give mean effects 11.9990 and 1.3692. Their difference is 10.6298, with exploratory interval [9.4417, 11.9641]. Native, identity and complete-restoration checks agree exactly; the algebraic preactivation reconstruction is accurate to 1.58e-5.

[Grouped head controls](DEVELOPMENT_HEAD_CONTROLS.json) nominate the widening pairs L1H0/L1H8 and L3H3/L3H11, using zero-based layer and head indices. Their joint replacement has factor-orbit mean effect 17.7148, compared with 4.6329 averaged over twenty random sets matched by layer and widening pairs. Key, value and query replacements give effects 7.8977, 7.2451 and 3.9252; order-bias replacement gives 0.0930. Joint query/key/bias replacement agrees exactly with attention-pattern replacement.

Ablating those four heads reduces pair accuracy from 5,436/5,445 (99.83%) to 3,926/5,445 (72.10%). Retaining only those heads gives 1,940/5,445 (35.63%), compared with 22.66% averaged over the matched random retained sets. Every MLP, residual path and embedding remains active. The selected heads contribute substantially but do not form a sufficient circuit.

[Two pruning studies](PRUNING_SUMMARY.json) distinguish exact choice fidelity from validity of a witness. The first permits no head-pair deletion under its registered prediction-agreement requirements, although 39 of 48 single-pair ablations retain at least 99% overall task accuracy. A separately registered pass preserving task correctness removes eighteen heads from the last two layers. Its [frozen mask](TASK_HEAD_PRUNING_SELECTION.json) retains 78 heads and all six MLPs; [evaluation on all twenty-seven criteria](TASK_HEAD_PRUNING_NATIVE_VALIDATION.json) gives minimum accuracy 99.129%. It solves 5,431/5,445 development pairs, but only 1,864/1,902 exposed S10 pairs (98.002%), compared with 1,874 for the unmodified model. This mask is an exploratory simplification, not an improvement in generalization or a minimum circuit. Apply it with `zeroed_heads` in [prune_development_task_heads.py](prune_development_task_heads.py) and the explicit forward function; the selected checkpoint itself remains unchanged.

[Parameter interventions during training](TRAINING_WEIGHT_REPLACEMENT.json) and [block-prefix replacements](TRAINING_PREFIX_REPLACEMENT.json) compare the earlier teacher-regularized model with the separately retained update-1,500 model. Transferring the first four updated blocks fixes 18 exposed S10 pairs without losses there; the complete update fixes twenty and loses one. A different development population regresses. These experiments localize changes in the learned parameters; they do not isolate the causal effects of the teacher penalty or new data.



The figure is also available as PDF and SVG. [FINAL_OUTCOME.json](FINAL_OUTCOME.json) collects the final results and their limitations.

## Earlier control-model experiments

The checkpoint is `artifacts/pinned_error_control_common5000.pt`, SHA-256 `63d89e2ccf6c10aaa8dc3d53c18f2e3607b19fb063c1768de92bb6ed418f0edb`. The matched sampler comparison selected this model over an arm using refreshed errors for the entire error allocation. Both were compared over the same predeclared first 5,000 updates. The original 6,000-update schedules reached their timed stopping conditions; their failures and the earlier common-window amendment remain in the record.

[Expanded interventions](EXPANDED_DECOMPOSITION.json) cover **3,803 difficult pairs in 357 factor orbits**, with both directions retained and no selection by correctness. Of these, 304 orbits were excluded from the earlier neuron selection. On those orbits:

| Intervention | Factor-group mean change in target contrast |
|---|---:|
| Fixed 16 hidden-unit pairs | 5.203 |
| Fixed 32 hidden-unit pairs | 8.310 |
| Fixed 64 hidden-unit pairs | 10.694 |
| Full first-MLP path | 11.889 |
| Fixed 16 pairs minus matched random sets | 3.644; exploratory 95% interval [3.212, 4.081] |

The path starts at non-candidate hidden units in the first MLP and enters the second attention layer through its keys and values. The fixed 32 pairs reproduce 69.89% of the mean full-path effect on these 304 orbits. Unit pairs follow the original widening indices \((k,k+384)\); a separate parameter audit finds that their present parameters are not exactly duplicated, so it does not produce a compressed checkpoint.

[Matched donors](SUPPORT_CONTROLS.json) give a further control on 1,247 inputs from 202 factor orbits. A donor with the same target support has the same ordered factors, candidate mask and positive supplied index, a different output permutation, and comparable inverse-permutation distance. Replacing the fixed 32 pairs with the opposite-support donor gives mean effect 9.684, versus 1.113 for the same-support donor. Their difference is 8.571, with exploratory group-bootstrap interval [7.609, 9.650].

An [exact preactivation decomposition](POSITION_DECOMPOSITION.json) separates the contributions of the inverse positions \(w^{-1}(j)\), \(w^{-1}(j+1)\), their order, rank columns, relative-root coordinates, attention and normalization. The position difference is the height of \(w^{-1}(\alpha_j)\). However, [a further intervention](ENDPOINT_RECEPTION.json) shows that most selected units favor one endpoint: retaining the larger of the two position coefficients in each unit gives mean effect 9.046, compared with 9.423 for both positions. Transmission through keys has effect 5.664, through values 3.286, and through both 8.310. These effects are not additive.



The figure is also retained as PDF and SVG. Its [caption and provenance](CIRCUIT_FIGURE.json) specify the populations and uncertainty calculation.

All pairs agree on the original candidate features. Only sixteen inputs from four factor orbits agree on the complete augmented candidate features. [An integer verification](AUGMENTED_FEATURE_OBSTRUCTION.md) also checks all candidate-to-candidate relations and exhibits disjoint valid witness sets. A selector using only these specified candidate data cannot solve both members. The larger intervention results therefore show use of non-candidate information; they do not show that this information is mathematically necessary for every pair in the cohort.

## Simplifications that failed

The [explicit root-endpoint attention rules](ROOT_ENDPOINT_ATTENTION.json) are inadequate. Restricting the two tested heads to simple roots with nonzero Cartan pairing against \((wv^{-1})(\alpha_j)\), while retaining their relative learned weights, reduces difficult-pair accuracy from 99.21% to 91.43%. A uniform distribution on that set performs worse.

A [fitted linear combination of inverse-position comparisons](ATTENTION_APPROXIMATION.json) explains 73.22% of centered attention-logit variation on factor orbits excluded from fitting and tuning. Its replacement gives 94.10% difficult-pair accuracy there, compared with 98.93% for the original model. This does not provide a faithful algebraic description of the attention computation.

Native and explicit forward passes, identity interventions and complete input restoration agree exactly. [The formal-study interface check](FORMAL_INTERFACE_CHECKS.json) also gives exact agreement of parameter gradients on development inputs. These implementation checks are separate from accuracy qualification and mathematical correctness.

## Mathematical deductions

[Levi restrictions](LEVI_RESTRICTIONS.md) proves that the Bruhat projection to a Levi subgroup commutes with downward Demazure operators in that subgroup. The resulting necessary length tests cannot exclude a candidate unless the tested Levi subset contains both the candidate index and the full simple-root support of the degree-one coefficient. All 51 retained errors are outside the reach of the tested consecutive blocks of semisimple rank at most three; the numerical tests correct none of them.

[Factorization over support components](SUPPORT_FACTORIZATION.md) is a corollary of the frozen primitive leading-form theorem. A nonzero equivariant Schubert coefficient factors, up to a positive integer, into primitive polynomials on the connected components of its simple-root support. The number of components is at most its degree. In degree one the support is therefore an interval. These deductions inherit the need for independent review of the manuscript's general proofs and make no external priority claim.

The [extended degree-one census](SECOND_NEIGHBOR_ROOT_SUPPORT_CHECK.json) finds no disconnected support among 11,442,479 retained exact entries. It includes 1,058,621 entries with nonunimodal coefficient sequences, so unimodality would be a false strengthening. Entries need not represent distinct coefficient polynomials. The [higher-degree check](SUPPORT_FACTORIZATION_CHECK.json) verifies the component factorization on 100 retained distinct examples, including fourteen with disconnected support. It is not a new exhaustive census.

The [all-Levi calculation](ALL_LEVI_LENGTH_TESTS.md) computes every necessary length restriction by downward Demazure operators and interval dynamic programming. It corrects six of thirty exposed S10 errors; twenty-four remain. [An explicit obstruction](LEVI_LENGTH_OBSTRUCTION.md) reduces to $a=152364$, $b=431256$, $x=461253$ in $S_6$: the supplied coefficient at index 4 is one, the candidate coefficient at index 3 is zero, and all Levi necessary length inequalities hold. Adjoining fixed points gives the same obstruction in every $S_N$, $N\ge6$. This deduction uses the frozen general theorems and retained exact coefficients, rather than a new independent $S_6$ product evaluation.

The [component reduction](TWO_COMPONENT_LEVI_REDUCTION.md) shows that in degree $d$ any failure of these necessary degree inequalities is detected by a Levi subgroup with at most $d+1$ components. In degree one the bound of two is sharp: a retained example passes all 24 connected tests containing its zero candidate but fails for the disconnected subset $\{4,6,7,8\}$.

[The ordinary support argument](LEVI_SUPPORT_OBSTRUCTION.md) strengthens the $S_6$ obstruction. Positivity and the Chevalley formula show that lowering a factor preserves existence of an ordinary nonzero output below a given Bruhat bound. Together with parabolic order and Demazure commutation, this implies that every proper Levi subgroup of $S_6$ containing candidate 3 passes the ordinary nonvanishing test. This concerns the stated existence test, not every possible use of local coefficient values.

## Reproduction and provenance

The experiment used Python 3.11, PyTorch 2.6.0 with CUDA 12.4, NumPy 2.4.6, lrcalc 2.1 and a local T4. `session.py` reads frozen helpers without redirecting their paths or ledgers. `full_resources.py` is the unique accounting module for this directory. Failure records, corrected orientation handling, launch errors and every charge are retained.

`restore_cache.py --verify-only` and `restore_second_cache.py --verify-only` check the two sets of retained archive streams. Without that flag it reconstructs missing compact data under `cache/data`; existing files must already have the expected hash. Earlier training sources use their own retained restoration scripts. Reconstruction makes no exact-product queries. New experimental runs need distinct names and must remain within the recorded resource agreement.

[The formal continuation plan](FORMAL_CONTINUATION_PLAN.md) was not executed because independent qualification failed. [STATE.md](STATE.md) records the final state and resumption constraints; [EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json) and [FINAL_RESOURCE_AUDIT.json](FINAL_RESOURCE_AUDIT.json) freeze the results and resource accounting.
