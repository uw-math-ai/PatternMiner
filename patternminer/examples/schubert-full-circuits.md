# Full training and circuit analysis

This continuation tests the workflow's interaction between model development, causal interventions, mathematical deductions, and failed independent assessments. The complete record is [2026-09-14-full-circuits](../../schubert-structure-constants/research/2026-09-14-full-circuits/README.md). The manuscript and earlier evidence remain frozen.

## Question and reliability requirement

For \(\ell(w)=\ell(u)+\ell(v)-1\), write
\[
A_i=c^w_{s_i u,v},\qquad B_j=c^w_{u,s_jv}.
\]
Given a positive \(A_i\), select a left descent \(j\) of \(v\) with \(B_j>0\). The model predicts from permutations, without coefficient labels as inputs. Its current architecture is independent of the supplied index. The requirement is at least 99% on each declared population, including every positive supplied index, both orientations, and simultaneous correctness on both members of difficult pairs.

The additional resource agreement permits four local T4 GPU-hours, six total timed hours and 20,000 exact-product queries, within the cumulative ledger and the existing storage limit. That authorization is retained and is not requested again at each step.

## Decisions and completed evidence

1. A matched comparison favors retaining some historical training errors in the sampling distribution. The two arms use the same predeclared comparison window. Their timed stopping conditions and the earlier schedule amendment are retained.
2. Exact products near development failures add 4,683,591 examples and 27,352 difficult pairs, with whole factor orbits excluded across training, development and assessment. The resulting 16,738,276-row training corpus produces a model passing all nineteen development criteria, with minimum accuracy 99.31%.
3. Independent assessment passes S7, S8 and S9, including all 323 difficult S9 pairs. It fails S10 difficult pairs at 1,278/1,308, despite 99.75% accuracy over individual supplied inputs. The assessment is explicitly reclassified as development; the original circuit discovery and confirmation populations remain protected.
4. A replacement assessment is reserved before adaptation. Further training retains supported predictions of an earlier model and adds 1,256,309 disjoint neighboring examples, bringing the corpus to 17,994,585 rows. Twenty-seven development criteria now include the exposed assessment. No assessment rows are added to training.
5. The larger corpus causes a memory allocation failure before the first optimizer update. Exact label range checks permit signed 16-bit storage, freeing about 1.73 GB without changing the batch size, targets or loss. The failure, its resource charge, and the correction are preserved.

The final model passes all twenty-seven development criteria, with minimum accuracy 99.303%. The replacement independent assessment passes S7, S8 and S9, but fails S10 difficult pairs at 1,874/1,902 = 98.528%, despite 99.799% accuracy on individual supplied inputs. The original formal discovery and confirmation populations remain unopened. The final development study actually measures every attention head and MLP on all 5,445 pairs; its causal conclusions are exploratory.

## What the circuit experiments establish

The completed expanded study uses 3,803 difficult pairs from 357 factor orbits on the earlier control model. Fixed pairs of hidden units in the first MLP contribute through the next attention layer's keys and values at non-candidate positions. On 304 orbits excluded from the earlier unit selection, the fixed 32 pairs reproduce 69.89% of the mean full-path contrast change. This is a ratio of intervention effects, not a fraction of the algorithm or accuracy explained.

Matched donors with the same target support produce much smaller effects than donors with disjoint target support. Exact preactivation decomposition identifies the inverse positions of the output permutation as the largest input contribution along this path. Further interventions show that many selected units favor one endpoint of the transformed simple root. These are controlled causal observations about a learned computation.

Two proposed simplifications fail. An explicit rule based on root endpoints reduces difficult-pair accuracy substantially. A fitted expression in inverse-position comparisons explains much of the attention-logit variance but also fails to preserve accuracy. The agent retains these failures instead of presenting the approximation as an algebraic algorithm.

Most pairs agree only on the original local candidate features. Only a small subset agrees on the complete augmented candidate features. The study therefore distinguishes use of non-candidate information from a proof that such information is necessary on every tested pair.

The final-model replay uses 5,445 pairs in 500 factor orbits. On 447 orbits excluded from the earlier neuron ranking, the fixed 32 hidden-unit pairs reproduce 67.35% of the mean full-path effect. The fixed 16-pair effect exceeds the matched random controls, and same-support donor controls again give much smaller effects than opposite-support donors. Native, identity and complete-restoration checks agree exactly. The independent accuracy failure prevents treating these results as formal confirmation.

The full head controls find that ablating four selected heads lowers development pair accuracy from 99.83% to 72.10%, while retaining only those heads gives 35.63%, with all MLPs and residual paths still active. They contribute strongly but are insufficient. Strict pruning by prediction identity removes no head pairs. A separately registered correctness-preserving pass removes eighteen heads and still passes all twenty-seven development criteria, but worsens exposed S10 pair accuracy to 98.002%. This distinction matters when several witness indices are valid: preserving the chosen index is a different objective from preserving correctness. Neither pruning result is formal confirmation.

## Mathematical consequences and a rejected strengthening

The [support-factorization deduction](../../schubert-structure-constants/research/2026-09-14-full-circuits/SUPPORT_FACTORIZATION.md) applies the frozen primitive leading-form theorem to the connected components of the simple-root support. In degree one, the support is an interval. The retained exact census contains many nonunimodal coefficient sequences, ruling out that stronger conjecture.

The remaining model errors motivate a second question: could the necessary length restrictions from all Levi subgroups characterize a valid candidate? Downward Demazure operators and an interval dynamic program make it possible to check every Levi subset. On the exposed errors, this corrects six of thirty predictions but leaves twenty-four zeros undetected.

The resulting [obstruction](../../schubert-structure-constants/research/2026-09-14-full-circuits/LEVI_LENGTH_OBSTRUCTION.md) reduces to
\[
a=152364,\qquad b=431256,\qquad x=461253.
\]
The written deduction gives
\[
q^x_{a,b}=\alpha_1+2\alpha_2+2\alpha_3+2\alpha_4+\alpha_5,
\quad c^x_{s_4a,b}=1,
\quad c^x_{a,s_3b}=0.
\]
Nevertheless candidate 3 satisfies every Levi necessary length inequality. Appending fixed points extends the example to every \(S_N\), \(N\ge6\). This refutes a natural proposed strengthening of the available restrictions. It does not rule out conditions using the actual Levi coefficients. The deduction depends on the frozen theorems; independent mathematical review and priority assessment remain separate tasks.

The [ordinary support argument](../../schubert-structure-constants/research/2026-09-14-full-circuits/LEVI_SUPPORT_OBSTRUCTION.md) strengthens the S6 obstruction further: every proper Levi subgroup containing the candidate passes the stated ordinary nonvanishing test, even when exact local coefficients are used. Positivity and the Chevalley formula supply the additional argument.

## What this tests about the workflow

The subsequent [component reduction](../../schubert-structure-constants/research/2026-09-14-full-circuits/TWO_COMPONENT_LEVI_REDUCTION.md) bounds the number of Levi components needed to detect a failure of these necessary degree inequalities by $d+1$ in degree $d$. The degree-one bound of two is sharp, with an explicit example that passes every connected test. A separate [representation obstruction](../../schubert-structure-constants/research/2026-09-14-full-circuits/AUGMENTED_FEATURE_OBSTRUCTION.md) verifies that all candidate features and candidate-to-candidate relations can agree while the valid witness sets are disjoint. This establishes a need for other positions on a specified subset, without identifying a unique neural circuit.

The initial question was not a supplied conjecture awaiting proof. Model errors and mechanistic interventions suggested which mathematical operations and examples to investigate. Exact data and mathematical reasoning then rejected an insufficient criterion and identified a smaller obstruction. Training strategy changed in response to independent failures, while the distinction between development and confirmation was retained.

This is one investigation performed by the workflow's developing agent. It supplies concrete examples of the research process, not an estimate of the workflow's effectiveness across mathematical problems.
