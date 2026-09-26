# Use mechanistic interpretability to generate mathematical conjectures

Read the mathematical question, model specification, data conventions, behavioral results, and resource agreement. Use the [method guide](../guides/mechanistic-interpretability.md) to select techniques relevant to the current uncertainty. The purpose is to obtain mathematical insight from the learned computation. A complete circuit reconstruction is one possible result, not a condition for proposing a conjecture.

Check behavioral reliability first. For extracting a dependable mathematical algorithm, require near-perfect correctness on the specified domain, including the difficult distinctions the proposed mechanism must explain. Use the threshold and assessment populations fixed in the model experiment; report the separate results, their sample sizes, and prior exposure. Agreement with the model is a different quantity from agreement with the mathematical target. If the model fails this requirement, return to improving its representation, training, or sampling. A bounded intervention aimed at diagnosing a particular failure can still be useful. High accuracy alone does not establish a mechanism, and imperfect accuracy does not invalidate an independently checked mathematical observation.

## State a hypothesis and its alternatives

Identify a specific behavior and explain how the model might compute it. Examples include representing an invariant, comparing two local quantities, selecting a position, or combining contributions. Distinguish the mathematical relation being investigated from the hypothesis about the model. List a plausible alternative when that would make an experiment more informative.

Check elementary explanations and known input symmetries first. Use valid mathematical examples, matched on relevant variables where possible. Specify what a pair of examples changes and what it preserves. If a perturbation leaves the space of valid objects, record that fact and limit the interpretation accordingly.

Check the model implementation before interpreting interventions. Fused inference paths may bypass hooks; a custom forward pass must reproduce the unmodified output, including masks, residual connections, normalization, and evaluation behavior. Record prior exposure of evaluation examples, and separate examples used to select a circuit from those used to assess the selected explanation at the appropriate mathematical grouping level.

Check identity interventions at each site separately and verify complete source restoration. Include every input-dependent path in that restoration, such as an attention bias computed directly from the mathematical input. If attribution uses gradients, check the differentiable forward and derivatives as well as inference outputs; an inference-only fused operator may be unsuitable. Preserve failed equivalence checks. Treat an intentional numerical or architectural change as a new model implementation and evaluate its behavior before interpreting it.

## Select and carry out useful tests

- **Direct inspection and probes:** examine weights and activations in a task-appropriate basis; test explicitly defined candidate statistics. Compare probe capacity and appropriate input, random, or untrained-model controls. A decoded statistic need not be used by the predictor.
- **Activation or path patching:** specify the source and destination, intervention site, alignment, replacement, downstream recomputation, and expected effect before execution. Check an identity intervention and relevant matched controls. Report raw output effects as well as any normalized measure.
- **Attention:** distinguish information transported by attention from the computation of the attention pattern. If selection is central, inspect query-key interactions and the effect of their interventions, including positional information and the softmax. Do not treat an attention heatmap as an algorithm.
- **Circuit analysis:** use approximate attribution to propose components or paths when useful, then check decisive claims by interventions. Evaluate retained behavior and unexplained contributions on specified examples; large weights or high circuit accuracy alone do not establish the proposed explanation.
- **Sparse representations:** use SAEs, transcoders, or cross-layer methods when they answer a question that direct analysis cannot. Check reconstruction, task behavior, residual contributions, and stability of the relevant distinction. Record the treatment of reconstruction error in every feature intervention.
- **Candidate algorithms:** compare the algorithm's variables and interventions with those of the model. Restrict and report the complexity of any learned alignment; test appropriate random-model and fresh-family controls. A flexible auxiliary model may learn the task itself.

These are options, not a sequence to complete. Use current primary guidance when adapting an advanced method. Confirm additional resources before expensive interpretation or an unapproved service. Pretrained interpretability tools and feature dictionaries are tied to their model, layer, and activation conventions; verify compatibility with the model under study.

## Interpret the evidence and formulate statements

Compare the recorded measurements with the pre-experiment prediction. Check metric direction, denominator, intervention implementation, and controls before writing an explanation. Distinguish evidence that a statistic is accessible, evidence that an intervention changes a behavior, and evidence for a particular computation. Negative and inconclusive results belong in the record.

Translate useful observations into standard mathematical objects: an invariant, inequality, recurrence, decomposition, construction, or other precise statement. Use [the conjecture prompt](conjecture.md) to assess its naturalness and content. Give the conjecture its own evidence and truth status. Neither fluent feature descriptions nor successful model interventions prove the mathematical claim. A partially resolved interpretation may still have suggested a valuable conjecture; preserve that contribution accurately.

Finish with the mathematical insight, the model explanation supported by the tests, the unresolved alternatives, and the next informative action. Do not spend the remaining project budget on a detailed explanation that no longer bears on a useful mathematical question.
