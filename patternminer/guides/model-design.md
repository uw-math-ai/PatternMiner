# Designing useful mathematical models

The recommendations here are starting choices for experiments. Adapt them to the mathematical objects, available evidence, and confirmed resources. The [source guide](sources.md) distinguishes published examples from general design proposals.

## Choose the task before the architecture

Write down what learning could expose that is difficult to inspect directly. Predicting an invariant can suggest a relation; learning a map can suggest an algorithm; ranking witnesses can expose a selection rule; generating exceptional examples can suggest a construction. A broad classification task is useful only if its predictions, errors, or representation could help formulate a better mathematical question.

Models trained on group operations provide a relevant precedent: Kvinge et al. find that some subgroup distinctions can be probed, with variation among runs and limits on other notions they can extract. Task accuracy alone therefore gives an incomplete picture of the structure available for investigation. [Small Algebraic Worlds](https://proceedings.mlr.press/v321/kvinge26a.html).

For a familiar map with exact labels, consider whether an alternative computation could be learned and extracted. For a conjectured relationship, distinguish testing that a relationship appears to exist from determining its exact form. For a known leading term or approximation, consider a residual target while retaining the original exact data and transformation.

## Match the representation to the objects

| Input structure | Reasonable candidates to compare | Principal question |
|---|---|---|
| A few scalar invariants | Linear or symbolic relations, boosted trees, small MLP | Is the nonlinear relationship useful beyond a simple formula? |
| Ordered words or sequences | Small encoder or encoder-decoder; recurrent computation when justified | Which positions, local operations, or iterations matter? |
| Unordered factors or components | Shared encoding and invariant aggregation, or an explicitly equivariant model | Which permutations preserve the target, and which information must remain distinct? |
| Graphs, posets, incidence structures | Graph models or structured attention with relevant incidence information | Can the representation distinguish the structures relevant to the question? |
| Candidate constructions or witnesses | Ranking or conditional generation with a mathematical validity check | Does the learned preference correspond to a useful rule? |

This table proposes comparisons rather than universal architecture recommendations. A graph model can miss distinctions the task requires; an invariant pooling operation can discard order that matters. Check such possibilities using small exact examples before interpreting training failure.

Tokenization and position conventions deserve an explicit experiment. The Abacus and Position Coupling studies show that encodings of task-relevant positions can change arithmetic length generalization. The transferable proposal is to expose justified structure and test extrapolation carefully, rather than assuming generic token positions suffice. Encoding a known operation also changes what can be claimed to have been discovered. [Abacus](https://arxiv.org/abs/2405.17399v1), [Position Coupling](https://arxiv.org/abs/2405.20671v1).

Preserve exact values for mathematical checking. If a target spans a large numerical range, compare appropriate normalization, separate zero/nonzero treatment, or a residual after a known expression. Any transformed prediction must have a clear interpretation in the original units. The enumerative-geometry study supplies a task-specific example, not a universal prescription for an activation function. [Enumerative geometry](https://arxiv.org/abs/2408.14915v3).

## Prevent the data from answering the wrong question

Define examples and symmetries before labels. Canonicalize independently of the answer. Keep duplicates, equivalent presentations, and recursive descendants with their generating family when the evaluation concerns new families. Document any larger examples used for training or tuning rather than calling them unseen extrapolation tests.

A target selected from many valid witnesses may encode an arbitrary oracle convention. Use set-valued supervision, ranking against verified alternatives, or a validity-based evaluation when feasible. Do not label another valid construction as mathematically wrong because it differs from one stored answer.

For rare events, record prevalence, family composition, and precision among examples selected for examination. A useful model may be a good selector without being a good global classifier. Preserve an appropriate random or elementary sampling comparison when claiming an improvement in discovery efficiency.

## Run a bounded diagnostic pilot

Estimate data-generation cost separately from training. Confirm the pilot's allocation. Verify labels and masking, compare the elementary baseline, and check whether a small training subset can be fitted. Examine the actual loss, valid-output rate, and errors by mathematical family.

If training fails, distinguish insufficient information, implementation mistakes, optimization, and inadequate capacity. If training succeeds but generalization fails, compare representations and sampling procedures before launching a large hyperparameter sweep. If both succeed, identify the pattern worth investigating mechanistically.

For extracting a reliable algorithm, adopt a project-specific requirement of near-perfect correctness before investing in circuit reconstruction. This is a workflow decision, not a claim that mechanistic methods require perfect models. Specify the mathematical domain and demanding assessment populations in advance. For witness selection, measure whether a valid witness is selected; binary accuracy over mostly invalid candidates can obscure failure. With matched examples having disjoint valid-output sets, report correctness on both members of each pair. Averages over many easy examples are insufficient for that purpose. Preserve difficult failures rather than removing them to satisfy the threshold.

Choose a model small enough to inspect while preserving the relevant behavior. The Dyck-path study provides an example of deliberate simplification; it does not make one-layer models mandatory. Save selected intermediate checkpoints when the evolution of a proposed mechanism is informative. The modular-addition grokking study motivates that option, without guaranteeing that prolonged training will resolve an unrelated problem. [Dyck paths](https://arxiv.org/abs/2605.30482v1), [Grokking](https://arxiv.org/abs/2301.05217v3).

## Prepare for interpretation and conjecturing

Retain the exact architecture, input and output conventions, seeds, checkpoint selection, and relevant data versions. Make weights, activations, attention outputs, and logits or regression outputs accessible. Decide whether a simple direct analysis could suffice before training an auxiliary interpreter.

Consider mechanistic interpretability whenever training is useful for pattern discovery. Ask which concrete hypothesis about the model would help formulate a conjecture and which experiment would distinguish it from alternatives. If predictions or selected examples already suggest a substantial mathematical statement, record it now. Interpretation can strengthen its motivation or reveal a different question later.

For every further training run, state the mathematical or methodological uncertainty it addresses. Improving a predictive score is useful when it improves the research instrument; it is not a prerequisite for reporting a good conjecture.
