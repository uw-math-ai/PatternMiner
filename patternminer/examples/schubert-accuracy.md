# Schubert continuation: improve accuracy before extracting an algorithm

The user requested near-perfect model accuracy before further mechanistic
interpretation. The preceding [circuit trial](schubert-trial.md) explained a
dependency in a model that was only 87.65% accurate on a historical sample
with inputs in `S_9`. More seriously, it got both answers right on only 25 of
458 matched pairs with disjoint valid witness sets. Reproducing that model's
computation would not yield a dependable witness-selection algorithm.

## Decision and experiment

The agent adopted a provisional 99% requirement on each assessment population,
including simultaneous correctness on the matched pairs. This threshold was
an operational choice made in response to the authors; it was not a claim about
the general prerequisites of mechanistic interpretability.

Within the remainder of the existing confirmed allocation, the agent compared
longer training with supervision of all candidate support labels and random
selection of a valid supplied witness. It then added `S_8` training examples.
Persistent errors on training data motivated a second controlled comparison:
additional training with the existing inputs versus explicit permutation rank
matrices, `r_z(p,q)=#{a<=p:z(a)<=q}`. All coefficient labels remained targets,
apart from the single certified supplied index in the task specification.

Complete unordered stable factor pairs were kept together. Validation selected
checkpoints; `S_9` examples were excluded from training and selection. Historical
assessment examples had already influenced the investigation, and the follow-up
was explicitly recorded as adaptive.

## Results and revision

| Assessment | Historical model | Selected rank-matrix model |
|---|---:|---:|
| Valid first witness, `S_7` inputs | 93.160% | 98.007% |
| Valid first witness, `S_8` inputs | 89.500% | 95.820% |
| Valid first witness, `S_9` inputs | 87.650% | 94.415% |
| Both members of each difficult pair | 25/458 | 180/458 |

The additional-training control reached 94.195% on `S_9`, so the rank features
provided only a small further improvement in this single-seed comparison.
The selected model still had thousands of training errors. No model met the
99% requirement, and no new circuit reconstruction was attempted.

The agent retained 4,096 records of confident invalid proposals from training
groups for a prospective hard-zero training comparison. Some records concern
different supplied witnesses for the same triple. They were mined from exact
historical labels and have not yet been used for retraining. A fresh evaluation
would still be necessary before claiming reliability.

The workflow revision asks the agent to establish near-perfect mathematical
correctness on a defined domain before extensive circuit analysis intended to
recover an algorithm. Limited interventions can still diagnose failures, and
an independently checked conjecture can remain valuable even when the model
that suggested it is imperfect. Mathematical correctness, fidelity to model
behavior, and a proposed mechanism remain separate assessments.

The [full experiment record](../../schubert-structure-constants/research/2026-09-13-witness-accuracy)
retains proposals, data membership and hashes, five checkpoints, scores,
verification, and cumulative resource consumption. The previous scientific
evidence and manuscript were preserved. This is another trial by the developing
agent, not independent validation of the workflow across mathematical domains.
