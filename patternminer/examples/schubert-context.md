# Schubert project context for resumption

Snapshot: 13 September 2026. Mathematical evidence is frozen at commit
`[withheld for anonymous review]`, tag
[withheld for anonymous review].
The chronology matters: the latest counterexample supersedes an open conjecture
in earlier notes. This context summarizes written results; it does not replace
their proofs or assert external verification of their novelty.

## Objective and conventions

Seek structural criteria, formulas, and constructions for Schubert structure
constants, particularly their vanishing. A natural conjecture explaining a
family of examples is useful even before a proof is available. Learned models
may suggest which mathematical operation or examples to investigate.

Use finite type A flag varieties over the complex numbers, with
`sigma_u = [X^u]` in codimension `ell(u)`. Write

\[
\sigma_u\sigma_v=\sum_w q^w_{u,v}\sigma_w,
\qquad \alpha_i=t_{i+1}-t_i.
\]

The ordinary coefficient is `c^w_{u,v}`; the equivariant coefficient has degree
`ell(u)+ell(v)-ell(w)`. Left multiplication by `s_i` exchanges **values**
`i,i+1`. A left descent is detected by `u^{-1}(i)>u^{-1}(i+1)`.
Distinguish stable ordinary multiplication from projection to a finite flag
variety. Appending a fixed point can matter when enumerating possible outputs.
Use the host manuscript's writing guide and established algebraic geometry
and algebraic combinatorics terminology.

For a set `J` of simple indices, write `w=xz`, where `x` belongs to the Levi
Weyl group `W_J` and `z` is minimal in the **left** coset `W_J w`. Put
`a=pi_z(u)`, `b=pi_z(v)`, where

\[
\pi_z(h)=\min\{p\in W_J:h\le pz\}.
\]

For a nonzero polynomial, `nu_J` is its least total degree in the variables
`alpha_j`, `j in J`, and `in_J` is its component of that degree.

## Current mathematical results

The following have written general proofs in the frozen record. External
mathematical review and priority assessment remain outstanding. Finite exact
checks support the implementations but do not establish the general statements.

1. **Positivity after a left descent.** If `s_i u<u<=w`, then
   \[
   q^w_{u,v}-\alpha_iq^w_{s_i u,v}
   \]
   is a nonnegative integral combination of products of distinct positive
   roots other than `alpha_i`. The stated ordinary/equivariant support
   relation and Conjectures A and B of Chen–Fan–Xiong–Yao follow, with the
   manuscript's finite and stable conventions. Equality
   `q^w_{u,v}=alpha_i q^w_{s_i u,v}` is equivalent to vanishing on
   `alpha_i=0`, under these hypotheses.
2. **Parabolic order of vanishing.** For `q^w_{u,v} != 0`,
   \[
   \nu_J(q^w_{u,v})=
   \min_{r\le x,\ q^{r,J}_{a,b}\ne0}
   \bigl(\ell(a)+\ell(b)-\ell(r)\bigr).
   \]
3. **Primitive leading form.** The nonzero Levi coefficients attaining
   this minimum have the same primitive part `P` (divide by the positive
   gcd of their coefficients). Then
   \[
   \operatorname{in}_J(q^w_{u,v})
   =P F,\qquad F\in\mathbb Z_{\ge0}[\alpha_k:k\notin J].
   \]
4. **Full Levi factor under local nonvanishing.** If also
   `q^{x,J}_{a,b} != 0`, the full Levi coefficient divides the initial form
   with a nonnegative integral quotient in the outside simple roots.
5. **Index of the generic Levi incidence fiber.** Let `B` be the local
   incidence image, `K=C(B)`, and `F_K` its proper integral generic
   incidence fiber. In general,
   \[
   \operatorname{in}_J(q^w_{u,v})
   =\operatorname{ind}(F_K)\operatorname{mdeg}(B)H,
   \qquad H\in\mathbb Z_{\ge0}[\alpha_k:k\notin J].
   \]
   The index is the gcd of degrees of **all** closed points of `F_K`.
   It need not equal the gcd of the distinguished Schubert degrees.

The outside factors have positive expressions in products of distinct outside
root coordinates before specialization; specialized weights may coincide.
All parabolic statements include disconnected `J` under their stated hypotheses.
They do not give a general positive combinatorial rule for ordinary constants.

Proofs: positivity and order,
primitive leading form,
full factor and index.

## A proposed strengthening is false

Take
\[
u=1325647,\quad v=1356724,\quad w=3456712,\quad J=\{1,2,3,4,5\}.
\]
Then `x=345612`, `z=1234756`, `a=132546`, `b=135624`. The global
ordinary coefficient is one. The unique minimum-degree nonzero Levi output
below `x` is `r=245613`, with coefficient two. Thus an integral expansion
in these specified minimum-degree coefficients would require `1=2F`.
It is impossible even with signed integral coefficients. The coefficient
at `x` is zero, so the full-factor theorem above does not apply.

The generic Levi fiber is a curve of index one: the additional coefficient
`c^{345612}_{132546,135642}=1` supplies a rational point. The distinguished
degree is two. Appending fixed points gives the same obstruction in every
`S_n`, `n>=7`. Monk's rule and the Grassmannian Pieri rule give an elementary
proof; independent localization and ordinary multiplication also check it.
See the counterexample.

Earlier work also proves a linear lower bound `H_n>=n-3` for the particular
family of subset-indexed projection tests defined there. A matching general
upper bound is not proved. Classical descent and rank restrictions remain
known baselines, not new theorems. The
claim ledger
records their scope and the status of all claims.

## Model evidence and a useful circuit question

The existing witness task uses degree-one equivariant examples. Given an
ordinary-support witness
\[
A_i=c^w_{s_i u,v}>0,
\]
rank left descents `j` of `v` for which `B_j=c^w_{u,s_j v}>0`.
Absent descents have coefficient zero by convention. Input features contain
permutations, indices, and lengths, but no coefficient values or target labels.
The supplied witness is information obtained from an exact computation.

A transformer with two layers, four attention heads per layer, and hidden
dimension 96 achieved 87.65% valid first proposals on 20,000 historical
examples with inputs from `S_9`. Stable outputs can belong to larger symmetric
groups. This is a sample, not an exhaustive test.
Both orientations and all outputs of each unordered stable input pair were
grouped for training separation. Three matched retrainings with supplied-witness
features removed reduced performance. Replacing the first valid supplied
witness with the last valid one also changed predictions, although the target
coefficients were unchanged. Individual head ablations did not identify a
complete mechanism.

The inequality motivating this task **predated training**. The geometric
theorems were not established by extracting a neural circuit. Historical
evaluation sets have already influenced research decisions; reusing them does
not constitute a new untouched holdout.

The subsequent [workflow trial](schubert-trial.md) found identical candidate
features with disjoint valid-descent sets, including an exact `S_7` obstruction
surviving the specified classical filters. It also localized most of the
model's margin difference on matched pairs to second-layer keys at noncandidate
tokens. These are additional trial results, separate from the frozen evidence
tag. They do not identify a complete combinatorial algorithm or change the
scope of the geometric theorems above.

The authors' next instruction makes behavioral reliability the immediate
priority. The historical model gets both witnesses right on only 25 of the
458 matched pairs with disjoint valid-descent sets. Further circuit work aimed
at recovering an algorithm should therefore wait for near-perfect correctness
on the intended domain. Five subsequent training continuations compared longer
training, supervision of the full witness support with valid supplied-witness
augmentation, larger training examples, and explicit permutation rank matrices.
The selected model reaches 94.415% on the historical `S_9` sample and gets both
answers correct on 180/458 matched pairs. It still fails the provisional 99%
requirement on each assessment population, including simultaneous correctness
on matched pairs. Those retained errors were used in the next continuation.
See the [accuracy continuation](schubert-accuracy.md). Any subsequent proposed
rule must be stated and checked as a mathematical assertion on its own terms.

The subsequent [preparation for circuit interpretation](schubert-readiness.md)
added fresh factor groups, diagram reflection, permutation-order attention
biases, and further layers. Its selected model reaches 99.746%, 98.890%, and
97.900% on the historical S7, S8, and S9 samples, respectively, and solves both
members of 342/458 difficult pairs. Fresh S9/S10 assessments over every valid
supplied witness reach 97.542%/97.690%; the full 99% requirement remains unmet.
The explicit intervention forward and gradients passed their implementation
checks. No new causal mechanism or recovered algorithm is claimed.

Mining also extended the information obstruction to candidate rank columns:
an independently checked S8 pair has identical specified data and supplied
coefficient one, but valid sets {4} and {2}. Appending fixed points preserves
the obstruction in every S_N with N at least eight. The manuscript and frozen
theorems above are unchanged. Resume from
`schubert-structure-constants/research/2026-09-13-circuit-readiness/STATE.md`; the next
target is reliability across new factor groups and valid supplied indices.

Historical evidence: [model and task](../../schubert-structure-constants/research/2026-09-08-theorem-search/witness_model.py),
head and witness interventions,
matched retraining.

## Resumption boundaries

The subsequent computational record is summarized in
[the full-scale continuation](schubert-full-circuits.md), with the live state in
[STATE.md](../../schubert-structure-constants/research/2026-09-14-full-circuits/STATE.md).
That continuation finishes training and a scan of all 96 heads and six MLPs on
5,445 development pairs. Its final model passes all 27 development criteria
(minimum 99.303%) but fails the replacement independent S10 pair assessment
(1,874/1,902 = 98.528%). Formal discovery and confirmation remain protected.
Grouped head ablation and retention show that the four nominated heads are
insufficient. A separate 78-head mask still passes all 27 development criteria
but worsens exposed S10 pair accuracy; it does not replace the selected model.
It adds deductions about factorization over connected components of root support,
a d+1 bound on Levi components for necessary degree tests, and an S6 obstruction
that also passes every proper Levi ordinary nonvanishing test containing its
candidate. The stated nonvanishing test is an existence condition; arbitrary
uses of local coefficient values are not ruled out.
These are later research notes; the frozen manuscript above remains unchanged.
Read the latest assessment and selection records before describing any model
as independently qualified. An earlier development pass was followed by an
independent S10 difficult-pair failure.

Keep the frozen manuscript and evidence unchanged. Write new experiments to
a fresh directory. Check actual availability of data and checkpoint files:
a source-code reference or Git hash does not imply the binary is in Git.
Read local mathematical references and reason from the evidence. Online
consultation is permitted for workflow methodology and technical documentation.
Obtain the project's resource agreement before running experiments, and retain
the agreement and consumption in the project record.
