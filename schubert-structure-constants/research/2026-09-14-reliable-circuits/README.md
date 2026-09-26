# Schubert witness selection and mechanistic analysis

This continuation aims to meet the specified 99% behavioral requirement and
then analyze the learned computation by causal interventions. The requirement
has **not yet been met**. The latest bounded runs are complete; see STATE.md for the pending resource extension and next investigation.
The earlier manuscript and frozen evidence have not been changed.

Given permutations with
\(\ell(u)+\ell(v)=\ell(w)+1\) and a supplied index \(i\) such that
\(c^w_{s_i u,v}>0\), the model must select a left descent \(j\) of \(v\)
with \(c^w_{u,s_jv}>0\). Every positive coefficient is an acceptable
answer. Inference receives the permutations and the single supplied index;
coefficient values and other positive indices are training labels only.

Assessment checks every valid supplied index, both orientations, and both
members of difficult pairs. Each assessment population must pass separately.
The first assessment failed on difficult pairs: 198/207 in S9 and 249/270
in S10, requiring both members to be correct for every valid supplied index.
Its predictions have now been inspected and its difficult pairs are explicitly
development data; see [ASSESSMENT_EXPOSURE.json](ASSESSMENT_EXPOSURE.json).
The circuit-discovery and circuit-confirmation populations remain unopened.
A replacement assessment must be fixed before its predictions are inspected.

The latest completed training run uses 6,552,106 exact rows, including new
factor pairs chosen from neighborhoods of development errors, and
augmentation by common fixed points.
It did not meet the expanded development requirement.
[NEIGHBOR_DATA.json](NEIGHBOR_DATA.json) records their construction and exact
checks. [RESOURCE_USE.json](RESOURCE_USE.json) records cumulative completed
charges; pending records and the live trainer must also be accounted for.

Several exact observations have been obtained during development:

- A certified incidence equation that
  is a difference of two flag minors, determined by exact rational incidence
  points and a rank certificate.
- Coefficient inequalities and
  weight-polytope inequalities, with
  internal proof drafts and exact finite checks.
- An obstruction to determining a witness from the equivariant coefficient
  and the full vector of coefficients on the first side,
  verified independently by divided differences.
- An obstruction involving two omitted rank-matrix columns,
  also independently verified.

External priority for these observations has not been investigated. The proof
drafts are separate from the frozen manuscript.

Development circuit measurements already
show causal contributions from noncandidate positions through the first two
transformer blocks. These measurements concern an imperfect earlier model.
They do not establish a general Schubert multiplication algorithm or satisfy
the requested behavioral requirement. Formal discovery and confirmation will
use the qualified checkpoint and separate populations.

Interventions under common fixed-point embeddings
provide an additional development diagnosis: 26 of 29 exposed incorrect
inputs admit an embedding on which the same model answers correctly. Source
embeddings were selected using development labels. These interventions do
not define a label-independent prediction method or constitute confirmation.

Storage maintenance records obsolete checkpoint
removal, verified data deduplication, and relocation of caches to `/mnt/data`.
No volume was formatted. Checkpoints and exact source artifacts used in
scientific claims remain available with recorded hashes.

A differential certificate for positive structure constants uses the incidence map. It succeeded on all 477 exposed difficult pairs in a separate geometric computation; those answers are excluded from neural accuracy.

Final development replay confirms that the last continuation remains below the requirement. Fixed predictor averages cannot close the gap on these populations. NEXT_RESEARCH.md gives the next representation and data comparison.
