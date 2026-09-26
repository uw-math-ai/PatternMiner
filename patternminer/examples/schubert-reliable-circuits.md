# A failed independent assessment and controlled circuit diagnosis

The requested task was to obtain a Schubert witness selector with at least
99% accuracy on each specified population, including every valid supplied
index and both members of difficult pairs, and then analyze its computation.
The [research record](../../schubert-structure-constants/research/2026-09-14-reliable-circuits/README.md)
contains the plans, exact data, checkpoints, numerical results, and mathematical
notes. The accuracy requirement remains unmet.

The agent expanded training to 6,552,106 exactly labelled examples, mined
21,776 difficult training pairs, added auxiliary ordinary and equivariant
coefficient targets, and tested augmentations from Poincare duality and
common fixed-point embeddings. One checkpoint passed all original development
criteria. Its first independent assessment passed on ordinary triples but
failed on difficult pairs: 198/207 in S9 and 249/270 in S10. Difficult training
pairs were much easier for this model, with 19,234/19,245 answered correctly
for every supplied index. These measurements exposed a generalization gap.

After the failed assessment informed strategy, its role was changed to
development data. The original predictions and checkpoint remained frozen;
its labels were not added to training. The evaluator now prevents a later
pass on those populations from being presented as independent qualification.
The separate circuit-discovery and circuit-confirmation populations remain
unopened, and a replacement assessment has not yet been carried out.

A bounded causal investigation used mathematically equivalent inputs obtained
by adjoining common initial and terminal fixed points. Twenty-six of 29
exposed incorrect inputs admitted an embedding on which the same model
answered correctly. Sources were selected using development labels, which is
explicitly disclosed. On 21 inputs with compatible controls, transferring the
aligned hidden state after two transformer blocks corrected 80.95%, compared
with 35.48% for shuffled sources from different factor groups. Individual MLP
interventions were less consistent across groups. These are causal measurements
of an imperfect model; they do not identify a Schubert multiplication algorithm.

Fixed prediction averages were tested separately without using labels to
choose an embedding at inference. They failed the 99% requirement. A finite
obstruction also rules out repairing the gap by convex averaging of the four
specified existing predictors: in six S9 pairs, an input has the same incorrect
candidate strictly above every valid target in all four predictors. Their
pair accuracy is consequently at most 201/207 on this population.

The accompanying mathematics includes a certified incidence equation that is
a difference of two flag minors, exact counterexamples to proposed sufficient
statistics, and a differential certificate for a positive ordinary structure
constant. The latter calculation finds valid witnesses on all 477 exposed
difficult pairs. It is a geometric computation and is excluded from neural
accuracy. Existing randomized nonvanishing results remain prior work; no new
complexity result or neural computation of the incidence equation is claimed.

This iteration illustrates why the workflow records unsuccessful assessment,
tracks exposure, tests causal specificity, and evaluates mathematical findings
separately from model performance. The next proposed representation uses
positive distinguished subexpressions for the two Richardson varieties.
A substantial further investigation requires the pending resource extension.
